"""Las cifras del artículo de métricas.

Todo el atlas termina cada sección con un número. Esta página pregunta qué
significa cada uno de esos números, y lo hace de la única forma que no admite
discusión: **enumerando**. Todas las matrices de confusión de tamaño cien están
escritas aquí, así que "estas dos métricas ordenan distinto" no es una opinión
sobre casos raros, es un recuento.

Tres familias, y las tres tienen el mismo defecto por el mismo motivo: resumir.

  1. **El área bajo la curva ROC es un estadístico de orden**, y se comprueba
     como identidad: el área trapezoidal y el recuento de pares concordantes de
     Mann y Whitney tienen que dar el mismo número, no uno parecido.
  2. **Por eso no ve la calibración.** Cualquier transformación monótona de las
     puntuaciones la deja intacta y mueve el Brier y el ECE, que es exactamente
     lo que el artículo 11 mide desde el otro lado. Aquí se aplica y se
     publican las tres columnas.
  3. **Y por eso tampoco ve la prevalencia**, mientras la precisión media sí:
     submuestrear negativos deja el área quieta y hunde la otra, con la
     referencia del azar (que es 0,5 siempre para una y la prevalencia para la
     otra) dibujada al lado.
  4. **El umbral no viene con la métrica.** Sobre las mismas puntuaciones, el
     umbral que maximiza exactitud, F1, MCC y el índice de Youden son cuatro
     umbrales distintos, y en un problema desequilibrado el de la exactitud
     puede ser el que no predice ni un positivo.
  5. **F1 y MCC, enumerados.** Con las 176.851 matrices de cien casos: cuántos
     pares ordenan al revés, cuánto llega a valer el desacuerdo, qué le pasa a
     cada una al intercambiar las clases, y la demostración de que F1 no mira
     los verdaderos negativos.
  6. **BLEU, medido sobre lo que no ve.** Perturbaciones controladas de frases
     reales del Brown (barajar, invertir, quitar, alargar, sustituir por
     vecinos de los vectores del artículo 46) y las dos cosas que casi nadie
     dice: que BLEU de corpus no es la media de los BLEU de frase, y que
     cambiar la segmentación mueve la cifra tanto como cambiar el sistema.

Ejecutar desde cualquier sitio: `python src/utils/generate_metrics_data.py`.
"""
import json
import os
import pickle
import re
import sys
from collections import Counter
from math import exp, log
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nlp_data import brown, reuters, tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "metrics" / "data"
CACHE = (Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
         / "metrics_cache")
EMBED = ROOT / "embeddings" / "data" / "embeddings.json"

SEED = 19
R8 = ("acq", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade")
DIMS = 60
ENUM_N = 100
BINS = 10
PREVALENCES = (0.5, 0.25, 0.1, 0.05, 0.02, 0.01)
BLEU_SENTENCES = 300
BLEU_MIN_LEN = 8

TAG = f"d{DIMS}n{ENUM_N}b{BINS}s{BLEU_SENTENCES}v{SEED}"


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


# ------------------------------------------------------------- las dos rutas
def auc_trapezoid(y, s):
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.concatenate([[0], np.cumsum(y == 1)])
    fp = np.concatenate([[0], np.cumsum(y == 0)])
    P, N = tp[-1], fp[-1]
    if P == 0 or N == 0:
        return None
    return float(np.trapezoid(tp / P, fp / N))


def auc_pairs(y, s):
    """El recuento de Mann y Whitney, con los empates valiendo medio par."""
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    R = ranks[y == 1].sum()
    n1, n0 = len(pos), len(neg)
    return float((R - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, s):
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y == 1)
    prec = tp / np.arange(1, len(y) + 1)
    P = tp[-1]
    return float((prec * (y == 1)).sum() / P) if P else None


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def logloss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, bins=BINS):
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for a, b in zip(edges, edges[1:]):
        m = (p > a) & (p <= b) if a > 0 else (p >= a) & (p <= b)
        if m.sum() == 0:
            continue
        tot += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(tot)


def confusion(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def f1(tp, fp, fn, tn):
    return 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def mcc(tp, fp, fn, tn):
    num = tp * tn - fp * fn
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else float(num / den)


def accuracy(tp, fp, fn, tn):
    return (tp + tn) / (tp + fp + fn + tn)


def youden(tp, fp, fn, tn):
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return tpr - fpr


# --------------------------------------------------------------- los datos
def reuters_table():
    docs, cats, split = reuters()
    keep = [i for i, c in enumerate(cats) if len(c) == 1 and c[0] in R8]
    texts = [tokenize(docs[i]) for i in keep]
    lab = np.array([R8.index(cats[i][0]) for i in keep])
    is_test = np.array([split[i] == "test" for i in keep])
    vocab = Counter()
    for t in texts:
        vocab.update(set(t))
    words = sorted([w for w, c in vocab.items() if c >= 5])
    idx = {w: k for k, w in enumerate(words)}
    X = np.zeros((len(texts), len(words)), np.float32)
    for i, t in enumerate(texts):
        for w in t:
            j = idx.get(w)
            if j is not None:
                X[i, j] += 1.0
    df = (X > 0).sum(0)
    X = np.log1p(X) * np.log(len(texts) / np.maximum(df, 1)).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    U, S, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    Z = (U[:, :DIMS] * S[:DIMS]).astype(np.float64)
    return Z, lab, is_test, len(words)


def stage_scores():
    """Cuatro modelos sobre la misma tarea binaria, con sus puntuaciones."""
    Z, lab, is_test, vocab = cached("reuters", reuters_table)
    out = dict(vocab=int(vocab), n=int(len(lab)), classes=list(R8),
               counts={c: int((lab == i).sum()) for i, c in enumerate(R8)})
    tasks = {}
    for name in ("earn", "ship"):
        c = R8.index(name)
        y = (lab == c).astype(int)
        tr, te = ~is_test, is_test
        models = {}
        for tag, mk in (("logistic", lambda: LogisticRegression(max_iter=3000)),
                        ("naive bayes", GaussianNB),
                        ("forest", lambda: RandomForestClassifier(
                            n_estimators=200, random_state=SEED, n_jobs=1)),
                        ("15 neighbours", lambda: KNeighborsClassifier(15))):
            m = mk()
            m.fit(Z[tr], y[tr])
            p = m.predict_proba(Z[te])[:, 1]
            models[tag] = dict(p=p.tolist(), y=y[te].tolist())
        tasks[name] = dict(models=models, prevalence=float(y[te].mean()),
                           n_test=int(te.sum()), positives=int(y[te].sum()))
    out["tasks"] = tasks
    return out


def metric_block(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    a, b = auc_trapezoid(y, p), auc_pairs(y, p)
    return dict(auc=a, auc_pairs=b, auc_gap=abs(a - b),
                ap=average_precision(y, p), brier=brier(y, p),
                logloss=logloss(y, p), ece=ece(y, p),
                acc_half=float(np.mean((p >= 0.5).astype(int) == y)))


# ------------------------------------------------------ 2. transformar monótono
def squash(p, k):
    """Comprime hacia 0,5 sin cambiar un solo orden."""
    z = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1 - 1e-9))
    return 1.0 / (1.0 + np.exp(-z / k))


def stage_monotone(y, p):
    rows = []
    base = metric_block(y, p)
    for k in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        q = squash(p, k)
        m = metric_block(y, q)
        rows.append(dict(k=k, auc=m["auc"], brier=m["brier"], ece=m["ece"],
                         logloss=m["logloss"],
                         auc_shift=abs(m["auc"] - base["auc"])))
    # y una monótona que no es una compresión: elevar a una potencia
    for g in (0.3, 3.0):
        q = np.clip(p, 1e-9, 1) ** g
        m = metric_block(y, q)
        rows.append(dict(k=None, power=g, auc=m["auc"], brier=m["brier"],
                         ece=m["ece"], logloss=m["logloss"],
                         auc_shift=abs(m["auc"] - base["auc"])))
    return dict(base=base, rows=rows,
                worst_shift=max(x["auc_shift"] for x in rows))


# ------------------------------------------------------------- 3. la prevalencia
def stage_prevalence(y, p, seeds=40):
    y = np.asarray(y)
    p = np.asarray(p)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rows = []
    for target in PREVALENCES:
        need = int(round(len(pos) * (1 - target) / target))
        if need > len(neg):
            continue
        aucs, aps = [], []
        for s in range(seeds):
            rng = np.random.default_rng(SEED + s)
            take = np.concatenate([pos, rng.choice(neg, need, replace=False)])
            aucs.append(auc_trapezoid(y[take], p[take]))
            aps.append(average_precision(y[take], p[take]))
        rows.append(dict(prevalence=target, n=int(len(pos) + need),
                         auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs)),
                         ap=float(np.mean(aps)), ap_sd=float(np.std(aps)),
                         ap_chance=target, auc_chance=0.5,
                         lift=float(np.mean(aps)) / target))
    return dict(rows=rows, seeds=seeds, positives=int(len(pos)))


# ---------------------------------------------------------------- 4. el umbral
def stage_threshold(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    ths = np.unique(np.concatenate([[0.0], np.sort(p), [1.0]]))
    if len(ths) > 2000:
        ths = np.quantile(ths, np.linspace(0, 1, 2000))
    best = {}
    curve = []
    for t in ths:
        cm = confusion(y, (p >= t).astype(int))
        vals = dict(accuracy=accuracy(*cm), f1=f1(*cm), mcc=mcc(*cm),
                    youden=youden(*cm))
        curve.append(dict(t=float(t), **vals))
        for k, v in vals.items():
            if k not in best or v > best[k][1]:
                best[k] = (float(t), float(v), cm)
    rows = []
    for k, (t, v, cm) in best.items():
        rows.append(dict(metric=k, threshold=t, value=v,
                         tp=cm[0], fp=cm[1], fn=cm[2], tn=cm[3],
                         predicted_positive=cm[0] + cm[1]))
    # la referencia tonta: no predecir ni un positivo
    cm0 = confusion(y, np.zeros_like(y))
    thin = curve[:: max(1, len(curve) // 300)]
    return dict(rows=rows, curve=thin, always_negative=dict(
        accuracy=accuracy(*cm0), f1=f1(*cm0), mcc=mcc(*cm0), youden=youden(*cm0)),
        prevalence=float(y.mean()))


# ------------------------------------------------------------ 5. la enumeración
def stage_enumerate(n=ENUM_N):
    """Todas las matrices de confusión con n casos, y qué opinan las métricas."""
    rows = []
    for tp in range(n + 1):
        for fp in range(n + 1 - tp):
            for fn in range(n + 1 - tp - fp):
                tn = n - tp - fp - fn
                rows.append((tp, fp, fn, tn))
    A = np.array(rows, dtype=np.int64)
    tp, fp, fn, tn = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        F = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        den = np.sqrt((tp + fp).astype(float) * (tp + fn) * (tn + fp) * (tn + fn))
        M = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0.0)
        ACC = (tp + tn) / n
    # F1 no mira los verdaderos negativos: parejas con el mismo tp, fp, fn
    same = {}
    for i in range(len(A)):
        key = (int(tp[i]), int(fp[i]), int(fn[i]))
        same.setdefault(key, []).append(i)
    worst_tn, worst_pair = 0.0, None
    for key, ids in same.items():
        if len(ids) < 2:
            continue
        lo, hi = ids[0], ids[-1]
        d = abs(M[hi] - M[lo])
        if d > worst_tn:
            worst_tn, worst_pair = float(d), (A[lo].tolist(), A[hi].tolist(),
                                              float(F[lo]), float(M[lo]), float(M[hi]))
    # desacuerdos de orden, muestreados sobre todos los pares (son demasiados
    # para enumerarlos, así que se sortean y se dice cuántos)
    rng = np.random.default_rng(SEED)
    pairs = rng.integers(0, len(A), (400000, 2))
    ok = pairs[:, 0] != pairs[:, 1]
    pairs = pairs[ok]
    df = F[pairs[:, 0]] - F[pairs[:, 1]]
    dm = M[pairs[:, 0]] - M[pairs[:, 1]]
    da = ACC[pairs[:, 0]] - ACC[pairs[:, 1]]
    flip_fm = float(np.mean((df * dm < 0)))
    flip_am = float(np.mean((da * dm < 0)))
    flip_af = float(np.mean((da * df < 0)))
    k = int(np.argmax(np.where(df * dm < 0, np.abs(df) + np.abs(dm), -1)))
    ex = (A[pairs[k, 0]].tolist(), A[pairs[k, 1]].tolist(),
          float(F[pairs[k, 0]]), float(F[pairs[k, 1]]),
          float(M[pairs[k, 0]]), float(M[pairs[k, 1]]))
    # simetría al intercambiar las clases
    swap = np.stack([tn, fn, fp, tp], 1)
    Fs = np.where(2 * swap[:, 0] + swap[:, 1] + swap[:, 2] > 0,
                  2 * swap[:, 0] / (2 * swap[:, 0] + swap[:, 1] + swap[:, 2]), 0.0)
    return dict(matrices=int(len(A)), n=n,
                flip_f1_mcc=flip_fm, flip_acc_mcc=flip_am, flip_acc_f1=flip_af,
                pairs=int(len(pairs)),
                worst_tn=worst_tn, worst_tn_case=worst_pair,
                worst_flip=ex,
                f1_asym=float(np.mean(np.abs(Fs - F))),
                f1_asym_max=float(np.max(np.abs(Fs - F))),
                mcc_asym=float(np.max(np.abs(M - M))),
                f1_all_negative=float(F[(tp == 0)].max()),
                cloud=[[int(a), int(b), float(c), float(d)]
                       for a, b, c, d in zip(tp[::97], fp[::97], F[::97], M[::97])])


# ------------------------------------------------------------------- 6. BLEU
def ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def bleu_stats(cand, ref, maxn=4):
    """Los recuentos con recorte, que es lo que suma un BLEU de corpus."""
    num, den = [], []
    for n in range(1, maxn + 1):
        c = ngrams(cand, n)
        rf = ngrams(ref, n)
        num.append(sum(min(v, rf[k]) for k, v in c.items()))
        den.append(max(0, len(cand) - n + 1))
    return num, den, len(cand), len(ref)


def bleu_from_stats(stats, maxn=4, smooth=False):
    num = [0] * maxn
    den = [0] * maxn
    cl = rl = 0
    for nu, de, c, rr in stats:
        for i in range(maxn):
            num[i] += nu[i]
            den[i] += de[i]
        cl += c
        rl += rr
    logs = []
    for i in range(maxn):
        a, b = num[i], den[i]
        if smooth:
            a, b = a + 1, b + 1
        if b == 0 or a == 0:
            return 0.0
        logs.append(log(a / b))
    bp = 1.0 if cl > rl else exp(1 - rl / max(cl, 1))
    return float(bp * exp(sum(logs) / maxn))


def sentence_bleu(cand, ref, maxn=4):
    return bleu_from_stats([bleu_stats(cand, ref, maxn)], maxn, smooth=True)


def load_vectors():
    d = json.loads(EMBED.read_text("utf-8"))
    v = d["vectors"]
    W = np.array(v["models"]["skip-gram"]["vec"], float)
    W = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    return v["words"], W, d["meta"]


def stage_bleu():
    sents = brown()
    rng = np.random.default_rng(SEED)
    pool = [s for s in sents if len(s) >= BLEU_MIN_LEN and len(s) <= 30]
    pick = rng.choice(len(pool), BLEU_SENTENCES, replace=False)
    refs = [[w.lower() for w, _ in pool[i]] for i in pick]
    words, W, emeta = load_vectors()
    wi = {w: k for k, w in enumerate(words)}
    sim = W @ W.T
    np.fill_diagonal(sim, -2.0)
    nearest = {w: words[int(np.argmax(sim[k]))] for w, k in wi.items()}
    vocab = sorted({w for s in refs for w in s})

    def perturb(kind, ref, g):
        s = list(ref)
        if kind == "identical":
            return s
        if kind == "shuffled":
            g.shuffle(s)
            return s
        if kind == "reversed":
            return s[::-1]
        if kind.startswith("drop"):
            k = int(kind.split()[1])
            if k >= len(s):
                return s[:1]
            keep = sorted(g.choice(len(s), len(s) - k, replace=False))
            return [s[i] for i in keep]
        if kind.startswith("swap"):
            k = int(kind.split()[1])
            for _ in range(k):
                if len(s) < 2:
                    break
                i = int(g.integers(0, len(s) - 1))
                s[i], s[i + 1] = s[i + 1], s[i]
            return s
        if kind.startswith("random"):
            k = int(kind.split()[1])
            for i in g.choice(len(s), min(k, len(s)), replace=False):
                s[i] = vocab[int(g.integers(0, len(vocab)))]
            return s
        if kind == "nearest neighbour":
            return [nearest.get(w, w) for w in s]
        if kind == "half":
            return s[:max(1, len(s) // 2)]
        if kind == "doubled":
            return s + s
        raise ValueError(kind)

    kinds = ("identical", "shuffled", "reversed", "drop 1", "drop 3",
             "swap 2", "random 2", "random 5", "nearest neighbour",
             "half", "doubled")
    rows = []
    for kind in kinds:
        g = np.random.default_rng(SEED + hash(kind) % 9999)
        stats, sent = [], []
        cov = []
        for ref in refs:
            cand = perturb(kind, ref, g)
            stats.append(bleu_stats(cand, ref))
            sent.append(sentence_bleu(cand, ref))
            cov.append(len(set(cand) & set(ref)) / len(set(ref)))
        st = bleu_from_stats(stats)
        num = [0] * 4
        den = [0] * 4
        for nu, de, _, _ in stats:
            for i in range(4):
                num[i] += nu[i]
                den[i] += de[i]
        rows.append(dict(kind=kind, corpus=st,
                         sentence_mean=float(np.mean(sent)),
                         precisions=[num[i] / den[i] if den[i] else 0.0
                                     for i in range(4)],
                         length=float(np.mean([len(perturb(kind, ref,
                                                           np.random.default_rng(SEED)))
                                               for ref in refs[:50]])),
                         word_overlap=float(np.mean(cov))))
    # la penalización por brevedad, en forma cerrada contra lo medido
    bp_rows = []
    for frac in (0.25, 0.5, 0.75, 0.9, 1.0, 1.25, 1.5):
        stats = []
        for ref in refs:
            k = max(1, int(round(len(ref) * frac)))
            cand = (ref * 3)[:k] if frac > 1 else ref[:k]
            stats.append(bleu_stats(cand, ref))
        cl = sum(s[2] for s in stats)
        rl = sum(s[3] for s in stats)
        bp_rows.append(dict(frac=frac, bleu=bleu_from_stats(stats),
                            bp=1.0 if cl > rl else float(exp(1 - rl / max(cl, 1))),
                            cand_len=int(cl), ref_len=int(rl)))
    # la segmentación: la misma pareja contada de tres maneras
    seg = []
    g = np.random.default_rng(SEED + 5)
    cands = [perturb("random 2", ref, g) for ref in refs]
    for tag, fn in (("as tokenised", lambda s: s),
                    ("punctuation glued", lambda s: [w for w in s if w.isalnum()]),
                    ("characters", lambda s: list(" ".join(s)))):
        stats = [bleu_stats(fn(c), fn(rf)) for c, rf in zip(cands, refs)]
        seg.append(dict(scheme=tag, bleu=bleu_from_stats(stats)))
    return dict(rows=rows, brevity=bp_rows, segmentation=seg,
                sentences=BLEU_SENTENCES,
                mean_ref_len=float(np.mean([len(s) for s in refs])),
                embed_meta=dict(corpus=emeta["corpus"], dim=emeta["dim"],
                                shipped=emeta["shipped"]))


def stage_disagree(tasks):
    """Cuántas veces dos métricas ordenan distinto a dos modelos de verdad."""
    rows = []
    for task, blob in tasks.items():
        names = list(blob["models"])
        vals = {}
        for nm in names:
            y = np.array(blob["models"][nm]["y"])
            p = np.array(blob["models"][nm]["p"])
            cm = confusion(y, (p >= 0.5).astype(int))
            vals[nm] = dict(auc=auc_trapezoid(y, p), ap=average_precision(y, p),
                            brier=-brier(y, p), f1=f1(*cm), mcc=mcc(*cm),
                            accuracy=accuracy(*cm))
        metrics = list(vals[names[0]])
        flips = 0
        total = 0
        detail = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                for a in range(len(metrics)):
                    for b in range(a + 1, len(metrics)):
                        da = vals[names[i]][metrics[a]] - vals[names[j]][metrics[a]]
                        db = vals[names[i]][metrics[b]] - vals[names[j]][metrics[b]]
                        total += 1
                        if da * db < 0:
                            flips += 1
                            detail.append(dict(a=names[i], b=names[j],
                                               m1=metrics[a], m2=metrics[b]))
        rows.append(dict(task=task, models=names, values=vals, flips=flips,
                         comparisons=total, detail=detail[:12]))
    return rows


def main():
    print("  1/6 los modelos y sus puntuaciones")
    sc = cached("scores", stage_scores)
    main_task = sc["tasks"]["ship"]
    y = np.array(main_task["models"]["logistic"]["y"])
    p = np.array(main_task["models"]["logistic"]["p"])

    print("  2/6 monótonas")
    mono = stage_monotone(y, p)
    print("  3/6 prevalencia")
    prev = stage_prevalence(y, p)
    print("  4/6 el umbral")
    th = stage_threshold(y, p)
    print("  5/6 la enumeración")
    enum = cached("enumerate", stage_enumerate)
    print("  6/6 BLEU")
    bl = cached("bleu", stage_bleu)
    dis = stage_disagree(sc["tasks"])

    blocks = {}
    for task, blob in sc["tasks"].items():
        blocks[task] = dict(prevalence=blob["prevalence"], n_test=blob["n_test"],
                            positives=blob["positives"],
                            models={nm: metric_block(np.array(d["y"]), np.array(d["p"]))
                                    for nm, d in blob["models"].items()})

    OUT.mkdir(parents=True, exist_ok=True)
    out = dict(
        meta=dict(seed=SEED, dims=DIMS, vocab=sc["vocab"], docs=sc["n"],
                  classes=list(R8), counts=sc["counts"], enum_n=ENUM_N,
                  bins=BINS, bleu_sentences=BLEU_SENTENCES,
                  bleu_min_len=BLEU_MIN_LEN, bleu_max_len=30,
                  note="the classifiers run on a singular value decomposition of "
                       "the same weighted counts the TF-IDF article measured"),
        tasks=r(blocks),
        monotone=r(mono),
        prevalence=r(prev),
        threshold=r(th),
        enumerate=r(enum),
        bleu=r(bl),
        disagree=r(dis),
        roc=dict(y=y.tolist(), p=r(p.tolist(), 5)),
    )
    path = OUT / "metrics.json"
    path.write_text(json.dumps(out, allow_nan=False), encoding="utf-8")
    print(f"  escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
