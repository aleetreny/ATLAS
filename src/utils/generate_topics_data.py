"""Las cifras del articulo de temas latentes (LDA).

Corre sobre **exactamente el mismo corpus** que el articulo de TF-IDF, y eso no
se declara, se afirma: el generador vuelve a construir las 7.790 noticias por
el mismo camino y comprueba contra el JSON ya publicado que el recuento de
documentos, el de tokens y el tamano del vocabulario coinciden. Sirve para dos
cosas. Una, que las dos paginas se puedan comparar cifra a cifra. Y dos, que
este articulo tenga algo que casi ningun articulo de topic modelling tiene: una
**clave de respuestas**. Las ocho categorias de Reuters existen, estan
etiquetadas a mano, y LDA no las ve. Asi que la pregunta "los temas que
encuentra son los temas que hay" deja de ser una cuestion de mirar las palabras
y pasa a tener un numero.

Dos implementaciones independientes del mismo modelo, que es la regla de la
casa: un muestreador de Gibbs colapsado escrito aqui (y reescrito en
JavaScript, porque el navegador lo ejecuta) y el variacional de scikit-learn.
No son el mismo algoritmo y no tienen por que dar el mismo optimo, asi que lo
que se compara entre ellos es lo que si esta definido: la verosimilitud que
alcanzan y las particiones que producen.

Ejecutar desde cualquier sitio: `python src/utils/generate_topics_data.py`.
"""
import base64
import json
import math
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nlp_data import reuters, stopwords, tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "topics" / "data"
CACHE = Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data")) / "topics_cache"
TFIDF_JSON = ROOT / "tf-idf" / "data" / "tfidf.json"

SEED = 19
R8 = ["earn", "acq", "crude", "trade", "money-fx", "interest", "money-supply", "ship"]


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


def b64_u16(a):
    return base64.b64encode(np.asarray(a, dtype="<u2").tobytes()).decode()


def b64_u32(a):
    return base64.b64encode(np.asarray(a, dtype="<u4").tobytes()).decode()


# --------------------------------------------------------------------- corpus
def load_corpus():
    docs, cats, split = reuters()
    keep = [(d, c[0], s) for d, c, s in zip(docs, cats, split)
            if len(c) == 1 and c[0] in R8]
    texts = [d for d, _, _ in keep]
    labels = [c for _, c, _ in keep]
    parts = [s for _, _, s in keep]
    toks = [tokenize(t) for t in texts]
    return texts, labels, parts, toks


def assert_same_corpus(toks, labels, parts):
    """Reutilizar los datos de otro articulo se demuestra, no se declara."""
    if not TFIDF_JSON.is_file():
        print("  (el JSON de tf-idf no esta, no se puede afirmar la identidad)")
        return None
    other = json.loads(TFIDF_JSON.read_text("utf-8"))
    m = other["meta"]
    mine = {"docs": len(toks), "train": sum(1 for p in parts if p == "train"),
            "tokens": sum(len(t) for t in toks)}
    theirs = {"docs": m["docs"], "train": m["train"], "tokens": other["zipf"]["tokens"]}
    for k in mine:
        assert mine[k] == theirs[k], f"{k}: {mine[k]} aqui contra {theirs[k]} en tf-idf"
    types = len({w for t in toks for w in t})
    assert types == other["zipf"]["types"], "el vocabulario no coincide"
    print(f"  identico al corpus de tf-idf: {mine['docs']} documentos, "
          f"{mine['tokens']} tokens, {types} tipos")
    return theirs


# ---------------------------------------------------------------- vocabulario
def build(toks, parts, min_df=5, max_df_frac=0.5, drop_stop=True, stop=None):
    """Matriz documento-palabra, con los dos cortes que todo el mundo aplica.

    `max_df` esta aqui porque es la otra mitad de la pregunta sobre palabras
    vacias: quitar una lista escrita a mano y quitar lo que sale en mas de la
    mitad de los documentos no son la misma operacion, y el articulo mide las
    dos por separado en vez de aplicar las dos y no saber cual hizo que.
    """
    df = Counter()
    for t in toks:
        df.update(set(t))
    n = len(toks)
    stop = set(stop or [])
    words = sorted(w for w, d in df.items()
                   if d >= min_df and d <= max_df_frac * n and not (drop_stop and w in stop))
    vocab = {w: i for i, w in enumerate(words)}
    rows = []
    for t in toks:
        c = Counter(vocab[w] for w in t if w in vocab)
        rows.append(sorted(c.items()))
    return words, vocab, rows


def to_csr(rows, n_words):
    from scipy.sparse import csr_matrix
    indptr = [0]
    idx, val = [], []
    for r in rows:
        for j, c in r:
            idx.append(j)
            val.append(c)
        indptr.append(len(idx))
    return csr_matrix((np.array(val, float), np.array(idx), np.array(indptr)),
                      shape=(len(rows), n_words))


# ------------------------------------------------------- Gibbs colapsado
def gibbs_lda(rows, n_words, k, alpha, beta, sweeps, seed, burn=None, count_every=5):
    """Muestreo de Gibbs colapsado, la version del articulo original.

    Se escribe a mano y no se toma de una libreria por una razon concreta: el
    navegador ejecuta este mismo bucle, token a token, y para que las dos
    orillas puedan compararse el generador aleatorio tiene que ser el mismo. El
    de numpy no existe en un navegador; un congruencial lineal si, y esta
    escrito identico a los dos lados.

    Devuelve las cuentas, no las distribuciones: las distribuciones se derivan
    de las cuentas con la misma formula en los dos sitios, asi que enviar
    cuentas enteras deja la comparacion exacta.
    """
    # Listas de Python y no arrays de numpy, por dos razones que van juntas.
    #
    # La que importa: `np.sum` sobre ocho elementos NO es una suma secuencial,
    # es un arbol (numpy suma por pares con ocho acumuladores a partir de n=8),
    # y un bucle de JavaScript si es secuencial. Dos ordenes de suma distintos
    # dan el ultimo bit distinto, y ese ultimo bit decide de que lado de `u`
    # cae el corte, asi que cada tantos miles de tokens los dos lados eligen
    # temas distintos y a partir de ahi las cadenas divergen sin que nada este
    # mal. Escrito como bucle a los dos lados, la comparacion vuelve a ser
    # exacta, que es lo unico que permite un guardia sobre enteros.
    #
    # La otra: sale unas cinco veces mas rapido, porque para k = 8 el coste de
    # numpy es todo overhead de llamada.
    state = seed & 0xFFFFFFFF

    def rnd():
        nonlocal state
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        return state / 4294967296.0

    # Un token por aparicion: LDA asigna un tema a cada aparicion, no a cada
    # tipo, asi que la matriz de conteos se despliega.
    doc_of, word_of = [], []
    for d, r in enumerate(rows):
        for j, c in r:
            for _ in range(c):
                doc_of.append(d)
                word_of.append(j)
    n_tok = len(doc_of)
    n_docs = len(rows)

    z = [0] * n_tok
    ndk = [[0] * k for _ in range(n_docs)]
    # Traspuesta respecto a la notacion del articulo (palabra por tema, no tema
    # por palabra) para que la fila que el bucle interno lee sea contigua.
    nwk = [[0] * k for _ in range(n_words)]
    nk = [0] * k
    for i in range(n_tok):
        t = int(rnd() * k)
        if t >= k:
            t = k - 1
        z[i] = t
        ndk[doc_of[i]][t] += 1
        nwk[word_of[i]][t] += 1
        nk[t] += 1

    wbeta = n_words * beta
    p = [0.0] * k
    trace = []
    for s in range(sweeps):
        for i in range(n_tok):
            d = doc_of[i]
            w = word_of[i]
            t = z[i]
            row_d = ndk[d]
            row_w = nwk[w]
            row_d[t] -= 1
            row_w[t] -= 1
            nk[t] -= 1
            tot = 0.0
            for c in range(k):
                v = (row_d[c] + alpha) * (row_w[c] + beta) / (nk[c] + wbeta)
                p[c] = v
                tot += v
            u = rnd() * tot
            acc = 0.0
            t = k - 1
            for c in range(k):
                acc += p[c]
                if u < acc:
                    t = c
                    break
            z[i] = t
            row_d[t] += 1
            row_w[t] += 1
            nk[t] += 1
        if (s + 1) % count_every == 0 or s == sweeps - 1:
            trace.append({"sweep": s + 1,
                          "loglik": float(joint_loglik(
                              np.array(ndk), np.array(nwk).T, np.array(nk),
                              alpha, beta, n_words))})
    return {"ndk": np.array(ndk), "nkw": np.array(nwk).T, "nk": np.array(nk),
            "z": np.array(z), "trace": trace, "n_tok": n_tok,
            "doc_of": np.array(doc_of), "word_of": np.array(word_of)}


def joint_loglik(ndk, nkw, nk, alpha, beta, n_words):
    """log p(w, z | alpha, beta), la cantidad que el muestreador sube.

    Es la unica cifra de progreso honesta para Gibbs: la verosimilitud de los
    datos sola no esta disponible sin integrar, y la perplejidad de train baja
    tambien cuando el modelo simplemente memoriza.
    """
    from scipy.special import gammaln
    k = ndk.shape[1]
    ll = 0.0
    ll += ndk.shape[0] * (gammaln(k * alpha) - k * gammaln(alpha))
    ll += float(np.sum(gammaln(ndk + alpha))) - float(np.sum(gammaln(ndk.sum(1) + k * alpha)))
    ll += k * (gammaln(n_words * beta) - n_words * gammaln(beta))
    ll += float(np.sum(gammaln(nkw + beta))) - float(np.sum(gammaln(nk + n_words * beta)))
    return ll


def phi_of(nkw, beta):
    return (nkw + beta) / (nkw + beta).sum(axis=1, keepdims=True)


def theta_of(ndk, alpha):
    return (ndk + alpha) / (ndk + alpha).sum(axis=1, keepdims=True)


# ----------------------------------------------------------------- coherencia
def npmi_coherence(top_words, rows, n_docs, n_words, topn=10):
    """NPMI media sobre los pares de las palabras de cabecera de cada tema.

    Coocurrencia a nivel de documento. El suelo de la medida (que casi nunca se
    publica) se estima con temas de palabras sacadas al azar del mismo
    vocabulario, porque "0,08 de coherencia" no significa nada sin saber que
    saca un tema que no es un tema.
    """
    bits = _doc_bitsets(rows, n_docs, n_words)
    df = [b.bit_count() for b in bits]

    def score(words):
        pairs = 0
        acc = 0.0
        for a in range(len(words)):
            for b in range(a + 1, len(words)):
                wa, wb = int(words[a]), int(words[b])
                co = (bits[wa] & bits[wb]).bit_count()
                pj = (co + 1e-12) / n_docs
                pa = df[wa] / n_docs
                pb = df[wb] / n_docs
                if pa <= 0 or pb <= 0:
                    continue
                acc += math.log(pj / (pa * pb)) / (-math.log(pj))
                pairs += 1
        return acc / pairs if pairs else 0.0

    per_topic = [score(t[:topn]) for t in top_words]
    return float(np.mean(per_topic)), [float(x) for x in per_topic]


_BITS_CACHE = {}


def _doc_bitsets(rows, n_docs, n_words):
    """Un entero de Python por palabra, con un bit por documento.

    La coocurrencia de dos palabras pasa a ser un AND y un popcount, que es el
    mismo truco que el articulo de cestas usa para los soportes. Sin esto, la
    coherencia de un barrido de nueve valores de k con su suelo por remuestreo
    son cientos de millones de pertenencias a conjuntos y la tirada no termina.
    """
    key = (id(rows), n_docs, n_words)
    if key in _BITS_CACHE:
        return _BITS_CACHE[key]
    bits = [0] * n_words
    for d, r in enumerate(rows):
        bit = 1 << d
        for j, _ in r:
            bits[j] |= bit
    _BITS_CACHE[key] = bits
    return bits


def coherence_floor(rows, n_docs, n_words, k, topn, seed=SEED, reps=10):
    rng = np.random.default_rng(seed)
    # Palabras al azar PONDERADAS por frecuencia de documento, no uniformes: un
    # tema de verdad tiene palabras frecuentes arriba, asi que un suelo hecho de
    # palabras uniformes compara contra algo mas raro que el objeto medido y
    # sale artificialmente bajo.
    df = np.zeros(n_words)
    for r in rows:
        for j, _ in r:
            df[j] += 1
    p = df / df.sum()
    vals = []
    for _ in range(reps):
        fake = [list(rng.choice(n_words, size=topn, replace=False, p=p)) for _ in range(k)]
        vals.append(npmi_coherence(fake, rows, n_docs, n_words, topn)[0])
    return float(np.mean(vals)), float(np.std(vals))


# ------------------------------------------------------------- contra la clave
def align_score(theta, labels):
    """Que tan bien la mezcla de temas recupera las categorias reales.

    Tres cifras porque miden cosas distintas y ninguna sola basta: el ARI de
    asignar cada documento a su tema dominante, la pureza de esa misma
    particion, y la exactitud del mejor emparejamiento uno a uno entre temas y
    categorias, que es la que responde a "hay un tema por categoria".
    """
    from sklearn.metrics import adjusted_rand_score
    from scipy.optimize import linear_sum_assignment
    hard = theta.argmax(axis=1)
    cats = sorted(set(labels))
    y = np.array([cats.index(l) for l in labels])
    k = theta.shape[1]
    M = np.zeros((k, len(cats)))
    for t, c in zip(hard, y):
        M[t, c] += 1
    purity = float(M.max(axis=1).sum() / len(y))
    r, c = linear_sum_assignment(-M)
    matched = float(M[r, c].sum() / len(y))
    return {"ari": float(adjusted_rand_score(y, hard)), "purity": purity,
            "one_to_one": matched, "topics_used": int((M.sum(axis=1) > 0).sum())}


def topic_match(phi_a, phi_b):
    """Emparejamiento optimo entre dos conjuntos de temas, y lo que se parecen.

    Un topic model no tiene orden: el tema 3 de una tirada no es el tema 3 de
    otra. Comparar sin emparejar mide la permutacion y nada mas, asi que se
    resuelve la asignacion optima por coseno y se publica la similitud del
    emparejamiento, que es la unica forma en que "los temas cambian" es una
    cifra.
    """
    from scipy.optimize import linear_sum_assignment
    A = phi_a / np.linalg.norm(phi_a, axis=1, keepdims=True)
    B = phi_b / np.linalg.norm(phi_b, axis=1, keepdims=True)
    S = A @ B.T
    r, c = linear_sum_assignment(-S)
    sims = S[r, c]
    return {"mean": float(sims.mean()), "min": float(sims.min()),
            "max": float(sims.max()), "sims": [float(x) for x in sorted(sims)]}


def jaccard_top(phi_a, phi_b, topn=10):
    from scipy.optimize import linear_sum_assignment
    A = phi_a / np.linalg.norm(phi_a, axis=1, keepdims=True)
    B = phi_b / np.linalg.norm(phi_b, axis=1, keepdims=True)
    r, c = linear_sum_assignment(-(A @ B.T))
    out = []
    for i, j in zip(r, c):
        sa = set(np.argsort(-phi_a[i])[:topn])
        sb = set(np.argsort(-phi_b[j])[:topn])
        out.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(out)), [float(x) for x in out]


def sk_lda(X, k, seed=SEED, max_iter=60):
    from sklearn.decomposition import LatentDirichletAllocation
    m = LatentDirichletAllocation(n_components=k, random_state=seed, max_iter=max_iter,
                                  learning_method="batch", doc_topic_prior=None,
                                  topic_word_prior=None)
    theta = m.fit_transform(X)
    theta = theta / theta.sum(axis=1, keepdims=True)
    phi = m.components_ / m.components_.sum(axis=1, keepdims=True)
    return m, theta, phi


def top_words_of(phi, words, n=10):
    return [[words[j] for j in np.argsort(-phi[t])[:n]] for t in range(phi.shape[0])]


def filtering_arms(toks, parts, labels, stop):
    """Que le pasa a un topic model cuando no se filtra nada.

    Tres brazos y **una sola cosa cambia entre cada par**, que es la unica
    forma de saber cual de los dos filtros hizo que. El brazo sin filtrar es el
    que casi nadie ensena, y es el que explica por que todo el mundo filtra.
    """
    out = []
    arms = [
        ("nothing removed", dict(drop_stop=False, max_df_frac=1.0)),
        ("words in over half the documents removed", dict(drop_stop=False, max_df_frac=0.5)),
        ("and the stopword list removed too", dict(drop_stop=True, max_df_frac=0.5)),
    ]
    stopset = set(stop)
    for name, kw in arms:
        w, v, r = build(toks, parts, min_df=5, stop=stop, **kw)
        X = to_csr(r, len(w))
        _, theta, phi = sk_lda(X, 8)
        tops = top_words_of(phi, w, 10)
        flat = [x for t in tops for x in t]
        coh, _ = npmi_coherence([[v[x] for x in t] for t in tops], r, len(r), len(w))
        al = align_score(theta, labels)
        out.append({
            "arm": name,
            "vocab": len(w),
            "tokens": int(sum(c for row in r for _, c in row)),
            "stopword_frac_in_top": sum(1 for x in flat if x in stopset) / len(flat),
            "coherence": coh,
            "ari": al["ari"],
            "purity": al["purity"],
            "topics": tops,
        })
        print(f"    {name:<44} {len(w):>5} words  "
              f"{out[-1]['stopword_frac_in_top']:.0%} stopwords in the headings  "
              f"coherence {coh:+.4f}  ARI {al['ari']:.4f}")
    return out


def k_sweep(X, rows, words, labels, ks=(2, 4, 6, 8, 10, 12, 16, 20, 30)):
    """Perplejidad retenida contra coherencia, que es el resultado del articulo.

    Las dos se calculan sobre la misma particion y el mismo ajuste. La
    perplejidad la da el propio scikit-learn desde su cota variacional sobre
    documentos que no vio; la coherencia sale de las palabras de cabecera
    contra la coocurrencia del corpus. Son dos criterios para elegir k y la
    pregunta es si eligen el mismo, no cual es mejor.
    """
    from sklearn.model_selection import train_test_split
    n = X.shape[0]
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED,
                              stratify=np.array(labels))
    floor_cache = {}
    out = []
    for k in ks:
        m, theta, phi = sk_lda(X[tr], k)
        pp = float(m.perplexity(X[te]))
        tops = top_words_of(phi, words, 10)
        vocab = {w: i for i, w in enumerate(words)}
        coh, per = npmi_coherence([[vocab[x] for x in t] for t in tops], rows, len(rows), len(words))
        if k not in floor_cache:
            floor_cache[k] = coherence_floor(rows, len(rows), len(words), k, 10)
        fl, fsd = floor_cache[k]
        al = align_score(m.transform(X) / m.transform(X).sum(axis=1, keepdims=True), labels)
        out.append({"k": k, "perplexity": pp, "coherence": coh, "coherence_floor": fl,
                    "coherence_floor_sd": fsd, "ari": al["ari"], "purity": al["purity"],
                    "topics_used": al["topics_used"], "topics": tops})
        print(f"    k={k:<3} perplexity {pp:9.2f}   coherence {coh:+.4f} "
              f"(floor {fl:+.4f})   ARI {al['ari']:.4f}")
    return {"rows": out, "n_train": int(len(tr)), "n_test": int(len(te))}


def stability(X, words, labels, seeds=(0, 1, 2, 3, 4), k=8):
    """Los temas no son estables entre semillas, y aqui es cuanto.

    Sin emparejar, comparar dos tiradas mide la permutacion; emparejadas por
    asignacion optima, lo que queda es la diferencia de verdad. Se dan las dos
    caras: el coseno entre distribuciones emparejadas (que ve el reparto
    entero) y el Jaccard de las diez palabras de cabecera (que es lo que un
    lector compara de hecho cuando mira dos tablas de temas).
    """
    fits = []
    for s in seeds:
        _, theta, phi = sk_lda(X, k, seed=s)
        fits.append((theta, phi))
    pairs = []
    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            m = topic_match(fits[i][1], fits[j][1])
            jac, _ = jaccard_top(fits[i][1], fits[j][1])
            pairs.append({"a": seeds[i], "b": seeds[j], "cosine": m["mean"],
                          "worst": m["min"], "jaccard": jac})
    aris = [align_score(t, labels)["ari"] for t, _ in fits]
    print(f"    coseno emparejado medio {np.mean([p['cosine'] for p in pairs]):.4f}, "
          f"peor tema {np.min([p['worst'] for p in pairs]):.4f}, "
          f"Jaccard de las diez cabeceras {np.mean([p['jaccard'] for p in pairs]):.4f}")
    print(f"    ARI contra la clave por semilla: "
          f"{', '.join(f'{a:.4f}' for a in aris)}")
    # Los dos primeros ajustes, ya EMPAREJADOS, para poder ponerlos uno al lado
    # del otro. Sin el emparejamiento, dos tablas de temas en orden crudo se
    # leen como si no se pareciesen en nada, que es una exageracion en la
    # direccion contraria: el modelo no tiene orden, asi que compararlos por
    # posicion mide la permutacion y no la diferencia.
    from scipy.optimize import linear_sum_assignment
    A = fits[0][1] / np.linalg.norm(fits[0][1], axis=1, keepdims=True)
    B = fits[1][1] / np.linalg.norm(fits[1][1], axis=1, keepdims=True)
    S = A @ B.T
    r, c = linear_sum_assignment(-S)
    ta = top_words_of(fits[0][1], words, 8)
    tb = top_words_of(fits[1][1], words, 8)
    matched = []
    for i, j in zip(r, c):
        sa, sb = set(ta[i]), set(tb[j])
        matched.append({"a": ta[i], "b": tb[j], "cosine": float(S[i, j]),
                        "shared": sorted(sa & sb),
                        "jaccard": len(sa & sb) / len(sa | sb)})
    matched.sort(key=lambda m: -m["cosine"])

    return {"pairs": pairs, "aris": [float(a) for a in aris],
            "seeds": list(seeds),
            "mean_cosine": float(np.mean([p["cosine"] for p in pairs])),
            "worst_topic": float(np.min([p["worst"] for p in pairs])),
            "mean_jaccard": float(np.mean([p["jaccard"] for p in pairs])),
            "ari_spread": float(np.std(aris)),
            "ari_min": float(np.min(aris)), "ari_max": float(np.max(aris)),
            "matched": matched}


def rivals(X, rows, words, labels, k=8):
    """LDA contra las dos factorizaciones que hacen el mismo trabajo.

    NMF y LSA ya salieron en el articulo de direcciones sobre otra matriz. Aqui
    corren sobre ESTA, con el mismo k, y se puntuan contra la misma clave y con
    la misma coherencia, que es la unica forma de que la comparacion diga algo.
    LSA se puntua sobre el valor absoluto de sus cargas porque sus componentes
    tienen signo y "las diez palabras de cabecera" no significa nada sin decir
    de que lado.
    """
    from sklearn.decomposition import NMF, TruncatedSVD
    from sklearn.feature_extraction.text import TfidfTransformer
    vocab = {w: i for i, w in enumerate(words)}
    out = []

    _, theta, phi = sk_lda(X, k)
    out.append(("LDA", theta, phi))

    T = TfidfTransformer(sublinear_tf=True).fit_transform(X)
    nm = NMF(n_components=k, random_state=SEED, init="nndsvda", max_iter=1000)
    W = nm.fit_transform(T)
    out.append(("NMF", W, nm.components_))

    sv = TruncatedSVD(n_components=k, random_state=SEED)
    S = sv.fit_transform(T)
    out.append(("LSA", np.abs(S), np.abs(sv.components_)))

    res = []
    for name, W, H in out:
        tops = top_words_of(H, words, 10)
        coh, _ = npmi_coherence([[vocab[x] for x in t] for t in tops], rows, len(rows), len(words))
        Wn = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
        al = align_score(Wn, labels)
        res.append({"model": name, "coherence": coh, "ari": al["ari"], "purity": al["purity"],
                    "one_to_one": al["one_to_one"], "topics_used": al["topics_used"],
                    "topics": tops})
        print(f"    {name:<5} coherence {coh:+.4f}   ARI {al['ari']:.4f}   "
              f"purity {al['purity']:.4f}   one-to-one {al['one_to_one']:.4f}")
    return res


def alpha_sweep_gibbs(rows, labels, k=8, alphas=(0.05, 0.5, 5.0, 50.0), sweeps=150):
    """El mismo barrido por encima del tope que impone la libreria.

    scikit-learn **rechaza** un `doc_topic_prior` mayor que 1: no lo avisa ni lo
    recorta, lanza. Es una restriccion de la implementacion y no del modelo (la
    Dirichlet esta definida para cualquier parametro positivo, y por encima de 1
    es justo donde deja de favorecer las esquinas del simplex y empieza a
    favorecer el centro), asi que el tramo alto se mide con el muestreador de
    esta pagina, que no tiene esa opinion. Se corre sobre el subconjunto para
    que cueste segundos, y se dice en la prosa que son dos protocolos distintos
    y por tanto que las dos mitades de la tabla no se restan entre si.
    """
    out = []
    n_words = 1 + max(j for r in rows for j, _ in r)
    for a in alphas:
        g = gibbs_lda(rows, n_words, k, a, 0.01, sweeps, SEED, count_every=sweeps)
        th = theta_of(g["ndk"], a)
        ent = -(th * np.log(np.maximum(th, 1e-12))).sum(axis=1)
        al = align_score(th, labels)
        out.append({"alpha": a, "mean_entropy": float(ent.mean()),
                    "max_entropy": float(np.log(k)),
                    "frac_dominant": float((th.max(axis=1) > 0.9).mean()),
                    "ari": al["ari"], "purity": al["purity"]})
        print(f"    (Gibbs) alpha={a:<6} entropia media {ent.mean():.4f} de {np.log(k):.4f}   "
              f"{out[-1]['frac_dominant']:.1%} con un tema al 90%   ARI {al['ari']:.4f}")
    return {"rows": out, "sweeps": sweeps, "docs": len(rows)}


def alpha_sweep(X, labels, k=8, alphas=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)):
    """El prior no es un detalle de implementacion, decide la forma del reparto.

    Se mide lo que alpha controla de verdad, que es cuanta masa se concentra en
    un tema por documento, y se pone al lado lo que eso le hace al marcador
    contra la clave. Muchas implementaciones lo fijan a 1/k y no lo dicen.
    """
    from sklearn.decomposition import LatentDirichletAllocation
    out = []
    for a in alphas:
        m = LatentDirichletAllocation(n_components=k, random_state=SEED, max_iter=60,
                                      learning_method="batch", doc_topic_prior=a)
        th = m.fit_transform(X)
        th = th / th.sum(axis=1, keepdims=True)
        ent = -(th * np.log(np.maximum(th, 1e-12))).sum(axis=1)
        al = align_score(th, labels)
        out.append({"alpha": a, "mean_entropy": float(ent.mean()),
                    "max_entropy": float(np.log(k)),
                    "frac_dominant": float((th.max(axis=1) > 0.9).mean()),
                    "ari": al["ari"], "purity": al["purity"]})
        print(f"    alpha={a:<6} entropia media {ent.mean():.4f} de {np.log(k):.4f}   "
              f"{out[-1]['frac_dominant']:.1%} de documentos con un tema al 90%   "
              f"ARI {al['ari']:.4f}")
    return out


def browser_subset(rows, words, labels, n_docs=600, min_count=3):
    """El corpus pequeno que el navegador muestrea en vivo, y su vocabulario.

    Se recorta a los documentos con suficientes palabras y a un vocabulario
    propio, porque lo que viaja tiene que caber y porque un muestreador que el
    lector ve correr necesita converger en segundos, no en minutos.
    """
    order = [i for i in range(len(rows)) if sum(c for _, c in rows[i]) >= 25]
    # Reparto equilibrado por categoria: con la proporcion natural, 600
    # documentos son 300 de "earn" y 4 de "ship", y el modelo no tiene de que
    # sacar ocho temas. El desequilibrio del corpus ya se mide en otro sitio.
    by = {}
    for i in order:
        by.setdefault(labels[i], []).append(i)
    per = n_docs // len(by)
    pick = []
    for c in sorted(by):
        pick.extend(by[c][:per])
    pick.sort()
    cnt = Counter()
    for i in pick:
        for j, c in rows[i]:
            cnt[j] += c
    keep = sorted(j for j, c in cnt.items() if c >= min_count)
    remap = {j: i for i, j in enumerate(keep)}
    sub = []
    for i in pick:
        r = [(remap[j], c) for j, c in rows[i] if j in remap]
        sub.append(sorted(r))
    return pick, [words[j] for j in keep], sub


def gibbs_vs_vb(rows, words, labels, k=8, sweeps=200):
    """Dos algoritmos para el mismo modelo, comparados en lo que si esta definido.

    Gibbs colapsado y ascenso variacional no optimizan la misma cosa y no tienen
    por que llegar al mismo sitio, asi que exigirles que coincidan tema a tema
    seria exigirles algo que el modelo no promete. Lo que si se puede comparar
    es lo que a un lector le importa: si encuentran los mismos temas (coseno del
    emparejamiento optimo), si separan igual de bien las categorias reales, y
    que coherencia alcanza cada uno.
    """
    vocab = {w: i for i, w in enumerate(words)}
    g = gibbs_lda(rows, len(words), k, 0.5, 0.01, sweeps, SEED, count_every=10)
    phi_g = phi_of(g["nkw"], 0.01)
    th_g = theta_of(g["ndk"], 0.5)
    X = to_csr(rows, len(words))
    _, th_v, phi_v = sk_lda(X, k)

    tg = top_words_of(phi_g, words, 10)
    tv = top_words_of(phi_v, words, 10)
    coh_g, _ = npmi_coherence([[vocab[x] for x in t] for t in tg], rows, len(rows), len(words))
    coh_v, _ = npmi_coherence([[vocab[x] for x in t] for t in tv], rows, len(rows), len(words))
    match = topic_match(phi_g, phi_v)
    jac, _ = jaccard_top(phi_g, phi_v)
    ag = align_score(th_g, labels)
    av = align_score(th_v, labels)
    print(f"    Gibbs   coherencia {coh_g:+.4f}  ARI {ag['ari']:.4f}")
    print(f"    varional coherencia {coh_v:+.4f}  ARI {av['ari']:.4f}")
    print(f"    coseno del emparejamiento {match['mean']:.4f}, peor tema {match['min']:.4f}, "
          f"Jaccard {jac:.4f}")
    return {"sweeps": sweeps, "gibbs": {"coherence": coh_g, **ag, "topics": tg},
            "vb": {"coherence": coh_v, **av, "topics": tv},
            "match": match, "jaccard": jac,
            "trace": g["trace"]}


def gibbs_refs(rows, n_words, k=8, alpha=0.5, beta=0.01, checkpoints=(1, 2, 5, 20)):
    """Estados exactos del muestreador para que el navegador se compare con ellos.

    Las cuentas son enteras y el generador aleatorio esta escrito igual a los
    dos lados, asi que esto no es una comparacion con tolerancia: o el navegador
    reproduce las mismas cuentas token a token o hay una diferencia real en la
    aritmetica. Se dan varios puntos de control en vez de solo el final para que
    el guardia pueda decir **en que barrido** empezaron a separarse, que es la
    diferencia entre un error localizable y un "no coincide".
    """
    out = []
    for cp in checkpoints:
        g = gibbs_lda(rows, n_words, k, alpha, beta, cp, SEED, count_every=max(1, cp))
        out.append({
            "sweeps": cp,
            "nk": [int(x) for x in g["nk"]],
            # La matriz entera de tema por palabra son 8 x 3.000 enteros y no
            # hace falta: un resumen por columnas la fija igual de bien y no
            # engorda el fichero. Si la suma por tema y la suma de cuadrados
            # coinciden en las ocho, la asignacion coincide.
            "nkw_rowsum": [int(x) for x in g["nkw"].sum(axis=1)],
            "nkw_sq": [int(x) for x in (g["nkw"].astype(np.int64) ** 2).sum(axis=1)],
            "ndk_sq": int((g["ndk"].astype(np.int64) ** 2).sum()),
            "loglik": g["trace"][-1]["loglik"],
            "first_z": [int(x) for x in g["z"][:40]],
        })
        print(f"    {cp} barrido(s): loglik {out[-1]['loglik']:.2f}")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("cargando el corpus")
    texts, labels, parts, toks = cached("corpus_r8", load_corpus)
    stop = stopwords()
    assert_same_corpus(toks, labels, parts)

    print("vocabulario")
    words, vocab, rows = cached("build", lambda: build(toks, parts, stop=stop))
    n_words = len(words)
    n_docs = len(rows)
    n_tok = sum(c for r in rows for _, c in r)
    print(f"  {n_words} palabras, {n_tok} tokens tras los cortes")

    X = to_csr(rows, n_words)

    print("los tres brazos de filtrado")
    filt = cached("filtering", lambda: filtering_arms(toks, parts, labels, stop))

    print("barrido de k: perplejidad contra coherencia")
    ks = cached("ksweep", lambda: k_sweep(X, rows, words, labels))

    print("estabilidad entre semillas")
    stab = cached("stability_v2", lambda: stability(X, words, labels))

    print("contra NMF y LSA sobre la misma matriz")
    riv = cached("rivals", lambda: rivals(X, rows, words, labels))

    print("el prior de documento")
    alp = cached("alpha_v2", lambda: alpha_sweep(X, labels))

    print("el subconjunto que corre en el navegador")
    pick, sub_words, sub_rows = cached(
        "subset", lambda: browser_subset(rows, words, labels))
    sub_labels = [labels[i] for i in pick]
    sub_titles = [(texts[i].strip().splitlines()[0] if texts[i].strip() else "")[:60]
                  for i in pick]
    print(f"  {len(sub_rows)} documentos, {len(sub_words)} palabras, "
          f"{sum(c for r in sub_rows for _, c in r)} tokens")

    print("el prior por encima del tope de la libreria, con el muestreador propio")
    alp_g = cached("alpha_gibbs",
                   lambda: alpha_sweep_gibbs(sub_rows, sub_labels))

    print("Gibbs propio contra el variacional, sobre ese subconjunto")
    gv = cached("gibbs_vs_vb",
                lambda: gibbs_vs_vb(sub_rows, sub_words, sub_labels))

    print("referencias exactas del muestreador para el navegador")
    refs = cached("gibbs_refs", lambda: gibbs_refs(sub_rows, len(sub_words)))

    data = {
        "meta": {
            "corpus": "the same 7,790 Reuters stories the tf-idf article uses",
            "docs": n_docs,
            "categories": R8,
            "category_counts": {c: labels.count(c) for c in R8},
            "vocab": n_words,
            "tokens": n_tok,
            "min_df": 5,
            "max_df_frac": 0.5,
            "seed": SEED,
            "stopwords_removed": len(stop),
        },
        "filtering": filt,
        "ksweep": ks,
        "stability": stab,
        "rivals": riv,
        "alpha": alp,
        "alpha_gibbs": alp_g,
        "gibbs_vs_vb": gv,
        "sampler": {
            "docs": len(sub_rows),
            "words": sub_words,
            "labels": sub_labels,
            "titles": sub_titles,
            "idx": b64_u16([j for r in sub_rows for j, _ in r]),
            "cnt": b64_u16([c for r in sub_rows for _, c in r]),
            "ptr": b64_u32(np.cumsum([0] + [len(r) for r in sub_rows])),
            "nnz": sum(len(r) for r in sub_rows),
            "tokens": sum(c for r in sub_rows for _, c in r),
            "alpha": 0.5, "beta": 0.01, "k": 8, "seed": SEED,
        },
        "refs": refs,
    }
    path = OUT / "topics.json"
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"escrito {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
