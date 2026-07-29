"""Las cifras del artículo de transferencia y destilación.

Las dos ideas de esta página tienen la misma forma: un modelo que ya existe hace
la mitad del trabajo del que viene después. En la transferencia lo que se hereda
son las **representaciones**; en la destilación, lo que se hereda son las
**respuestas**. Y las dos se venden con una explicación que casi nunca viene con
su control al lado, así que aquí lo que se mide no es si funcionan (funcionan)
sino contra qué.

El montaje son dos tablas de imágenes de 28x28 con diez clases cada una, MNIST y
Fashion-MNIST, en las dos direcciones. Las redes son perceptrones multicapa
escritos en numpy (`tinynet.py`), lo cual limita lo que se puede concluir y hay
que decirlo: aquí no hay convoluciones y por tanto no hay nada que decir sobre
qué transfiere una jerarquía visual. Lo que sí se puede decir es lo que la
literatura afirma sobre matrices de pesos y sobre distribuciones de salida, que
es de lo que va el artículo.

Lo que se mide:

  1. **La curva de transferencia con sus cuatro controles.** Contra los píxeles
     crudos, contra la MISMA arquitectura sin entrenar (que es el control que
     casi nadie corre y que decide si lo que transfiere es el entrenamiento o la
     forma), contra una red entrenada con las etiquetas **barajadas** y contra
     una entrenada sobre la propia tarea de destino, que es el techo.
  2. **Ajuste fino contra sonda congelada contra empezar de cero**, sobre los
     mismos datos y en el mismo barrido de tamaño, que es donde se ve que la
     ventaja de heredar se paga y se agota.
  3. **La destilación con el control que separa sus dos explicaciones.** Un
     estudiante entrenado con las probabilidades del maestro aprende dos cosas
     a la vez: qué clase es y cómo se reparte lo que sobra. Endurecer la
     salida del maestro deja solo la primera; **barajar la masa entre las
     clases equivocadas** deja solo la primera conservando la entropía, que es
     el control que dice si el conocimiento oscuro existe.
  4. **La temperatura, barrida, con suelo de semillas**, en vez de citada.
  5. **Y el truco original**: el maestro puede etiquetar datos que nadie
     etiquetó, y eso es lo que convierte la destilación en algo más que una
     regularización.

Ejecutar desde cualquier sitio: `python src/utils/generate_distillation_data.py`.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tinynet import Net, onehot, softmax, train  # noqa: E402
from vision_data import fashion_mnist, mnist  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "distillation" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "distillation_cache")

SEED = 19
HID = (256, 128)              # la arquitectura compartida por todos los brazos
SRC_N = 12000                 # filas de la tarea de origen
TGT_TEST = 5000               # filas de test de la tarea de destino
SRC_EPOCHS = 12
FT_EPOCHS = 20
SIZES = (10, 25, 50, 100, 250, 500, 1000, 2500, 5000)
SEEDS = 2

TEACHER_HID = (512, 512)
TEACHER_MEMBERS = 3
TEACHER_N = 12000
STUDENT_HID = (32,)
DISTIL_EPOCHS = 20
DISTIL_SEEDS = 3
TEMPS = (1.0, 2.0, 4.0, 8.0, 16.0)
UNLABELLED = 12000

TAG = (f"h{'-'.join(map(str, HID))}n{SRC_N}e{SRC_EPOCHS}f{FT_EPOCHS}"
       f"s{SEEDS}t{TEACHER_MEMBERS}d{DISTIL_EPOCHS}u{UNLABELLED}v{SEED}")


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


def load():
    a = mnist()
    b = fashion_mnist()
    return dict(mnist=(flat(a[0]), a[1], flat(a[2]), a[3]),
                fashion=(flat(b[0]), b[1], flat(b[2]), b[3]))


def stratified(y, n, rng):
    """n filas repartidas entre las clases lo más igual que se pueda."""
    classes = np.unique(y)
    per = n // len(classes)
    take = []
    for c in classes:
        idx = np.where(y == c)[0]
        take.extend(rng.choice(idx, min(per, len(idx)), replace=False).tolist())
    left = n - len(take)
    if left > 0:
        rest = np.setdiff1d(np.arange(len(y)), np.array(take))
        take.extend(rng.choice(rest, left, replace=False).tolist())
    return np.array(sorted(take))


# ---------------------------------------------------------------- codificadores
def make_encoders(src, seed):
    """Las tres redes de origen: la honrada, la de etiquetas barajadas, y nada.

    La tercera no se entrena y es el control que más importa: si una sonda
    sobre rasgos de una red **sin entrenar** ya sube mucho respecto a los
    píxeles, lo que la transferencia trae no es lo aprendido, es la proyección.
    """
    X, y, Xt, yt = src
    rng = np.random.default_rng(seed)
    idx = stratified(y, SRC_N, rng)
    out = {}
    honest = Net([X.shape[1], *HID, 10], np.random.default_rng(seed))
    train(honest, X[idx], onehot(y[idx], 10), np.random.default_rng(seed + 1),
          epochs=SRC_EPOCHS, batch=128, lr=1e-3)
    out["source"] = dict(net=honest, acc=honest.accuracy(Xt[:TGT_TEST], yt[:TGT_TEST]))

    ysh = y[idx].copy()
    np.random.default_rng(seed + 2).shuffle(ysh)
    shuffled = Net([X.shape[1], *HID, 10], np.random.default_rng(seed))
    train(shuffled, X[idx], onehot(ysh, 10), np.random.default_rng(seed + 3),
          epochs=SRC_EPOCHS, batch=128, lr=1e-3)
    out["shuffled"] = dict(net=shuffled,
                           train_acc=shuffled.accuracy(X[idx], ysh),
                           acc=shuffled.accuracy(Xt[:TGT_TEST], yt[:TGT_TEST]))

    out["random"] = dict(net=Net([X.shape[1], *HID, 10], np.random.default_rng(seed + 9)),
                         acc=None)
    return out


def probe(F, y, Ft, yt, seed):
    """La sonda: regresión logística sobre rasgos congelados."""
    m = LogisticRegression(max_iter=3000, C=1.0, random_state=seed)
    m.fit(F, y)
    return float(m.score(Ft, yt))


def stage_transfer(tables, src_name, tgt_name):
    src = tables[src_name]
    Xg, yg, Xgt, ygt = tables[tgt_name]
    Xgt, ygt = Xgt[:TGT_TEST], ygt[:TGT_TEST]
    enc = cached(f"encoders-{src_name}", lambda: make_encoders(src, SEED))
    oracle = cached(f"oracle-{tgt_name}", lambda: make_encoders(
        (Xg, yg, Xgt, ygt), SEED + 77)["source"])

    feats = {
        "pixels": (Xg, Xgt),
        "random": (enc["random"]["net"].features(Xg), enc["random"]["net"].features(Xgt)),
        "shuffled": (enc["shuffled"]["net"].features(Xg),
                     enc["shuffled"]["net"].features(Xgt)),
        "transfer": (enc["source"]["net"].features(Xg), enc["source"]["net"].features(Xgt)),
        "oracle": (oracle["net"].features(Xg), oracle["net"].features(Xgt)),
    }
    rows = []
    for n in SIZES:
        for s in range(SEEDS):
            rng = np.random.default_rng(SEED + 1000 * s + n)
            idx = stratified(yg, n, rng)
            row = dict(n=n, seed=s)
            for tag, (F, Ft) in feats.items():
                row[tag] = probe(F[idx], yg[idx], Ft, ygt, SEED + s)
            # ajuste fino y desde cero, sobre la misma muestra
            base = enc["source"]["net"].copy()
            base.W[-1] = np.zeros_like(base.W[-1])
            base.b[-1] = np.zeros_like(base.b[-1])
            train(base, Xg[idx], onehot(yg[idx], 10),
                  np.random.default_rng(SEED + s), epochs=FT_EPOCHS, batch=32, lr=3e-4)
            row["finetune"] = base.accuracy(Xgt, ygt)
            scratch = Net([Xg.shape[1], *HID, 10], np.random.default_rng(SEED + 31 * s))
            train(scratch, Xg[idx], onehot(yg[idx], 10),
                  np.random.default_rng(SEED + s), epochs=FT_EPOCHS, batch=32, lr=3e-4)
            row["scratch"] = scratch.accuracy(Xgt, ygt)
            rows.append(row)
        print(f"    {src_name} -> {tgt_name}: n = {n} hecho")
    return dict(rows=rows,
                source_acc=enc["source"]["acc"],
                shuffled_train=enc["shuffled"]["train_acc"],
                shuffled_acc=enc["shuffled"]["acc"],
                oracle_full=oracle["acc"])


def stage_freeze(tables, src_name, tgt_name, n=500):
    """Qué congelar: nada, la primera capa, o todo menos la cabeza."""
    src = tables[src_name]
    Xg, yg, Xgt, ygt = tables[tgt_name]
    Xgt, ygt = Xgt[:TGT_TEST], ygt[:TGT_TEST]
    enc = cached(f"encoders-{src_name}", lambda: make_encoders(src, SEED))
    rows = []
    for tag, live in (("head only", [2]), ("last two", [1, 2]), ("everything", [0, 1, 2])):
        accs = []
        for s in range(SEEDS):
            rng = np.random.default_rng(SEED + 1000 * s + n)
            idx = stratified(yg, n, rng)
            net = enc["source"]["net"].copy()
            net.W[-1] = np.zeros_like(net.W[-1])
            net.b[-1] = np.zeros_like(net.b[-1])
            train(net, Xg[idx], onehot(yg[idx], 10), np.random.default_rng(SEED + s),
                  epochs=FT_EPOCHS, batch=32, lr=3e-4, trainable=live)
            accs.append(net.accuracy(Xgt, ygt))
        trainable = int(sum(net.W[i].size + net.b[i].size for i in live))
        rows.append(dict(arm=tag, trainable=trainable, mean=float(np.mean(accs)),
                         sd=float(np.std(accs)), accs=accs))
    return dict(rows=rows, n=n, total=int(enc["source"]["net"].n_params()))


# ------------------------------------------------------------------ destilación
def teacher_probs(members, X, T=1.0, batch=4096):
    out = np.zeros((len(X), 10))
    for m in members:
        for i in range(0, len(X), batch):
            out[i:i + batch] += m.probs(X[i:i + batch], T=T)
    return out / len(members)


def stage_teacher(tgt):
    X, y, Xt, yt = tgt
    rng = np.random.default_rng(SEED)
    idx = stratified(y, TEACHER_N, rng)
    members, accs = [], []
    for k in range(TEACHER_MEMBERS):
        net = Net([X.shape[1], *TEACHER_HID, 10], np.random.default_rng(SEED + 100 + k))
        train(net, X[idx], onehot(y[idx], 10), np.random.default_rng(SEED + 200 + k),
              epochs=SRC_EPOCHS, batch=128, lr=1e-3)
        members.append(net)
        accs.append(net.accuracy(Xt[:TGT_TEST], yt[:TGT_TEST]))
        print(f"    maestro {k + 1}/{TEACHER_MEMBERS}: {accs[-1]:.4f}")
    P = teacher_probs(members, Xt[:TGT_TEST])
    return dict(members=members, idx=idx, member_acc=accs,
                ensemble_acc=float(np.mean(np.argmax(P, 1) == yt[:TGT_TEST])),
                params=int(members[0].n_params() * TEACHER_MEMBERS))


def scramble_wrong(P, rng):
    """Baraja la masa de las clases equivocadas fila a fila.

    Conserva la clase ganadora y su probabilidad, y conserva la entropía
    exactamente, así que un estudiante que aprenda igual de bien con esto no
    está aprendiendo nada de las semejanzas entre clases: está aprendiendo de
    cuánta confianza tiene el maestro.
    """
    Q = P.copy()
    top = np.argmax(P, 1)
    for i in range(len(P)):
        others = np.array([c for c in range(P.shape[1]) if c != top[i]])
        Q[i, others] = P[i, rng.permutation(others)]
    return Q


def stage_distil(tgt, teach):
    X, y, Xt, yt = tgt
    Xt, yt = Xt[:TGT_TEST], yt[:TGT_TEST]
    idx = teach["idx"]
    members = teach["members"]
    Xl, yl = X[idx], y[idx]
    Yhard = onehot(yl, 10)
    soft = {T: teacher_probs(members, Xl, T=T) for T in TEMPS}
    hardened = onehot(np.argmax(soft[1.0], 1), 10)
    scr = scramble_wrong(soft[4.0], np.random.default_rng(SEED))

    def student(Y, seed, T=1.0, X_=None, Yhard_=None, weight=1.0):
        net = Net([X.shape[1], *STUDENT_HID, 10], np.random.default_rng(seed))
        train(net, Xl if X_ is None else X_, Y, np.random.default_rng(seed + 1),
              epochs=DISTIL_EPOCHS, batch=128, lr=1e-3, T=T,
              soft_weight=weight, Yhard=Yhard_)
        return net.accuracy(Xt, yt)

    arms = {}

    def run(tag, fn):
        accs = [fn(SEED + 7 * s) for s in range(DISTIL_SEEDS)]
        arms[tag] = dict(mean=float(np.mean(accs)), sd=float(np.std(accs)), accs=accs)
        print(f"    {tag}: {np.mean(accs):.4f}")

    run("hard labels", lambda s: student(Yhard, s))
    for T in TEMPS:
        run(f"soft T={T:g}", lambda s, T=T: student(soft[T], s, T=T))
    run("teacher argmax", lambda s: student(hardened, s))
    run("scrambled wrong classes", lambda s: student(scr, s, T=4.0))

    # el techo del estudiante: la misma red con todas las etiquetas de verdad
    run("student ceiling", lambda s: student(onehot(y[:TEACHER_N + UNLABELLED], 10),
                                             s, X_=X[:TEACHER_N + UNLABELLED]))

    # etiquetar datos que nadie etiquetó
    extra = np.arange(TEACHER_N, TEACHER_N + UNLABELLED)
    Xu = np.concatenate([Xl, X[extra]])
    Su = np.concatenate([soft[4.0], teacher_probs(members, X[extra], T=4.0)])
    run("soft T=4 plus unlabelled", lambda s: student(Su, s, T=4.0, X_=Xu))

    mixes = []
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        accs = [student(soft[4.0], SEED + 7 * s, T=4.0, Yhard_=Yhard, weight=w)
                for s in range(DISTIL_SEEDS)]
        mixes.append(dict(weight=w, mean=float(np.mean(accs)), sd=float(np.std(accs))))
    entropies = {f"{T:g}": float(np.mean(-(soft[T] * np.log(np.clip(soft[T], 1e-12, None)))
                                          .sum(1))) for T in TEMPS}
    student_params = int(Net([X.shape[1], *STUDENT_HID, 10],
                             np.random.default_rng(0)).n_params())
    return dict(arms=arms, mixes=mixes, entropies=entropies,
                scrambled_entropy=float(np.mean(-(scr * np.log(np.clip(scr, 1e-12, None)))
                                                .sum(1))),
                teacher_agree=float(np.mean(np.argmax(soft[1.0], 1) == yl)),
                student_params=student_params,
                teacher_params=teach["params"],
                compression=teach["params"] / student_params,
                labelled=int(len(idx)), unlabelled=UNLABELLED,
                seeds=DISTIL_SEEDS, temps=list(TEMPS))


def main():
    tables = load()
    print(f"  MNIST {tables['mnist'][0].shape}, Fashion {tables['fashion'][0].shape}")

    print("  1/4 transferencia MNIST -> Fashion")
    fwd = cached("transfer-mnist-fashion",
                 lambda: stage_transfer(tables, "mnist", "fashion"))
    print("  2/4 transferencia Fashion -> MNIST")
    bwd = cached("transfer-fashion-mnist",
                 lambda: stage_transfer(tables, "fashion", "mnist"))
    print("  3/4 qué congelar")
    frz = cached("freeze", lambda: stage_freeze(tables, "mnist", "fashion"))

    print("  4/4 destilación sobre Fashion")
    teach = cached("teacher", lambda: stage_teacher(tables["fashion"]))
    dist = cached("distil", lambda: stage_distil(tables["fashion"], teach))

    def summarise(d):
        rows = d["rows"]
        arms = [k for k in rows[0] if k not in ("n", "seed")]
        out = []
        for n in SIZES:
            sub = [row for row in rows if row["n"] == n]
            rec = dict(n=n)
            for a in arms:
                v = [row[a] for row in sub]
                rec[a] = float(np.mean(v))
                rec[a + "_sd"] = float(np.std(v))
            out.append(rec)
        return dict(rows=out, arms=arms, source_acc=d["source_acc"],
                    shuffled_train=d["shuffled_train"], shuffled_acc=d["shuffled_acc"],
                    oracle_full=d["oracle_full"])

    OUT.mkdir(parents=True, exist_ok=True)
    out = dict(
        meta=dict(seed=SEED, hidden=list(HID), source_n=SRC_N, test_n=TGT_TEST,
                  source_epochs=SRC_EPOCHS, tune_epochs=FT_EPOCHS, seeds=SEEDS,
                  sizes=list(SIZES),
                  teacher_hidden=list(TEACHER_HID), teacher_members=TEACHER_MEMBERS,
                  student_hidden=list(STUDENT_HID), distil_epochs=DISTIL_EPOCHS,
                  note="plain multilayer perceptrons in numpy, no convolutions; "
                       "every number is an accuracy on the target test split"),
        transfer=dict(forward=r(summarise(fwd)), backward=r(summarise(bwd))),
        freeze=r(frz),
        teacher=dict(member_acc=r(teach["member_acc"]),
                     ensemble_acc=r(teach["ensemble_acc"]),
                     params=teach["params"], members=TEACHER_MEMBERS),
        distil=r(dist),
    )
    path = OUT / "distillation.json"
    path.write_text(json.dumps(out, allow_nan=False), encoding="utf-8")
    print(f"  escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
