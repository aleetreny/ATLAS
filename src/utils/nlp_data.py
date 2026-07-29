"""Los corpus de texto del modulo 4.1, fuera del repo y descargados una vez.

Mismo contrato que `vision_data.py`: nada de esto se commitea, vive en
`~/.atlas_vision_data/nltk_data/` y un clon limpio lo vuelve a bajar solo. Los
ficheros vienen del repositorio `nltk/nltk_data` en GitHub, que es el unico
host de datasets que el proxy de esta maquina deja pasar.

Lo que expone, y por que ese y no otro:

  reuters()      10.788 noticias de Reuters-21578 con sus categorias. Es el
                 corpus de recuperacion y clasificacion del modulo: tiene
                 etiquetas, tiene documentos de longitud muy desigual (de 12 a
                 2.700 palabras) y esa desigualdad es justo lo que el articulo
                 de TF-IDF necesita para que la normalizacion signifique algo.
  brown()        1,16 millones de palabras etiquetadas con categoria
                 gramatical. Es la unica fuente de verdad de este modulo para
                 una etiqueta por token, asi que el HMM se puede puntuar y los
                 vecinos de un embedding se pueden preguntar por su categoria.
  gutenberg()    dieciocho libros completos. El corpus de modelado de lenguaje,
                 porque un libro tiene dependencias largas de verdad.
  movie_reviews() 2.000 criticas con polaridad, mitad y mitad exactas.
  stopwords()    la lista inglesa, que este modulo usa para MEDIR que quitarla
                 no hace lo que la gente cree, no para aplicarla a ciegas.

Todas las funciones cachean en disco y devuelven estructuras de Python planas,
sin depender de que nltk este importado en el proceso que llama.
"""
import hashlib
import io
import os
import re
import urllib.request
import zipfile
from pathlib import Path

CACHE = Path(os.environ.get("ATLAS_DATA_DIR", Path.home() / ".atlas_vision_data"))
NLTK = CACHE / "nltk_data"
BASE = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages"

# Los digests son de los zips que esta maquina bajo la primera vez. No son un
# control de seguridad, son un control de reproducibilidad: si GitHub sirve otro
# fichero, las cifras del modulo dejan de ser las que se publicaron y hay que
# enterarse en la descarga y no tres articulos despues.
PACKAGES = {
    "reuters": ("corpora/reuters.zip", "9a59a43823f02a6e"),
    "brown": ("corpora/brown.zip", "9b275f9b3b95d7bd"),
    "gutenberg": ("corpora/gutenberg.zip", "2d3c3ab548c65394"),
    "movie_reviews": ("corpora/movie_reviews.zip", "a41211ae68501913"),
    "stopwords": ("corpora/stopwords.zip", "48c0e52d8b52546e"),
    "universal_tagset": ("taggers/universal_tagset.zip", "d490e1ae8f5625dc"),
}


def _fetch(name):
    """Baja y descomprime un paquete de nltk_data si no esta ya en el cache."""
    target = NLTK / "corpora" / name
    alt = NLTK / "taggers" / name
    if target.is_dir() or alt.is_dir():
        return
    rel, want = PACKAGES[name]
    NLTK.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{rel}"
    print(f"  bajando {name} de {url}")
    with urllib.request.urlopen(url, timeout=180) as r:
        blob = r.read()
    digest = hashlib.sha256(blob).hexdigest()[:16]
    print(f"  {name}: {len(blob) / 1024 / 1024:.1f} MB, sha256[:16] {digest}")
    if digest != want:
        raise SystemExit(
            f"{name} llego con digest {digest} y las cifras publicadas se midieron "
            f"sobre {want}. Antes de seguir hay que decidir si se republican.")
    out = NLTK / rel.split("/")[0]
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(out)


def _read(path, encoding="latin-1"):
    return path.read_text(encoding, errors="replace")


# --------------------------------------------------------------------- reuters
def reuters():
    """(docs, cats, split) para las 10.788 noticias de Reuters-21578.

    docs   lista de strings, el cuerpo entero incluida la primera linea de
           titular, que es como lo entrega nltk.
    cats   lista de listas de categorias (una noticia puede tener varias).
    split  lista de "train"/"test", la particion ModApte que trae el corpus.

    El orden es el de `cats` en el fichero del corpus, que es alfabetico por id,
    asi que es estable entre maquinas: nada de aqui depende de como ordene el
    sistema de ficheros.
    """
    _fetch("reuters")
    root = NLTK / "corpora" / "reuters"
    docs, cats, split = [], [], []
    for line in _read(root / "cats.txt").splitlines():
        parts = line.split()
        if not parts:
            continue
        fid, labels = parts[0], parts[1:]
        body = _read(root / fid)
        docs.append(body)
        cats.append(labels)
        split.append("train" if fid.startswith("training/") else "test")
    return docs, cats, split


# ----------------------------------------------------------------------- brown
BROWN_UNIVERSAL = None


def _universal_map():
    """El mapa de las 472 etiquetas finas del Brown a las doce universales."""
    global BROWN_UNIVERSAL
    if BROWN_UNIVERSAL is not None:
        return BROWN_UNIVERSAL
    _fetch("universal_tagset")
    path = NLTK / "taggers" / "universal_tagset" / "en-brown.map"
    m = {}
    for line in _read(path, "utf-8").splitlines():
        if not line.strip():
            continue
        fine, coarse = line.split("\t")[:2]
        m[fine] = coarse
    BROWN_UNIVERSAL = m
    return m


def brown(universal=True):
    """Frases del Brown como listas de (palabra, etiqueta).

    Con `universal=True` las etiquetas son las doce del tagset universal, que es
    lo que hace comparable un HMM aqui con cualquier cifra publicada. Las finas
    del Brown son 472 y con ellas la matriz de transicion tiene 222.784 celdas
    para 57.340 frases: no es que salga mal, es que no se puede estimar.
    """
    _fetch("brown")
    root = NLTK / "corpora" / "brown"
    umap = _universal_map() if universal else None
    sents = []
    for f in sorted(root.iterdir()):
        if not re.fullmatch(r"c[a-r]\d\d", f.name):
            continue
        for line in _read(f).splitlines():
            line = line.strip()
            if not line:
                continue
            sent = []
            for tok in line.split():
                word, _, tag = tok.rpartition("/")
                if not word:
                    continue
                tag = tag.upper()
                if umap is not None:
                    # Las etiquetas compuestas del Brown ("NN+HVZ") y las
                    # negadas ("BEDZ*") no estan todas en el mapa; se cae al
                    # primer trozo, que es como lo resuelve el propio nltk.
                    tag = umap.get(tag) or umap.get(tag.split("+")[0].rstrip("*")) or "X"
                sent.append((word, tag))
            if sent:
                sents.append(sent)
    return sents


# ------------------------------------------------------------------ gutenberg
def gutenberg(fileid="austen-emma.txt"):
    """El texto crudo de uno de los dieciocho libros que trae el corpus."""
    _fetch("gutenberg")
    return _read(NLTK / "corpora" / "gutenberg" / fileid, "utf-8")


def gutenberg_files():
    _fetch("gutenberg")
    return sorted(p.name for p in (NLTK / "corpora" / "gutenberg").iterdir()
                  if p.suffix == ".txt")


# --------------------------------------------------------------- movie reviews
def movie_reviews():
    """(docs, labels) con 1.000 criticas positivas y 1.000 negativas."""
    _fetch("movie_reviews")
    root = NLTK / "corpora" / "movie_reviews"
    docs, labels = [], []
    for lab in ("neg", "pos"):
        for f in sorted((root / lab).iterdir()):
            if f.suffix != ".txt":
                continue
            docs.append(_read(f))
            labels.append(lab)
    return docs, labels


# ------------------------------------------------------------------- stopwords
def stopwords():
    _fetch("stopwords")
    return _read(NLTK / "corpora" / "stopwords" / "english", "utf-8").split()


# ----------------------------------------------------------------- tokenizador
# Un tokenizador, escrito una vez, usado por los siete articulos y **copiado
# palabra por palabra en JavaScript** en `assets/js/textkit.js`. Cualquier
# diferencia entre los dos lados mueve todas las cifras del modulo a la vez, asi
# que es deliberadamente tonto: minusculas y rachas de letras. Sin guiones, sin
# apostrofos, sin numeros. "don't" da ["don", "t"], y eso es una decision que el
# articulo cuenta, no un descuido.
TOKEN = re.compile(r"[a-z]+")


def tokenize(text):
    return TOKEN.findall(text.lower())


if __name__ == "__main__":
    docs, cats, split = reuters()
    print(f"reuters: {len(docs)} docs, "
          f"{sum(1 for s in split if s == 'train')} train, "
          f"{len({c for cs in cats for c in cs})} categorias")
    sents = brown()
    print(f"brown: {len(sents)} frases, {sum(len(s) for s in sents)} tokens, "
          f"{len({t for s in sents for _, t in s})} etiquetas universales")
    mr, lab = movie_reviews()
    print(f"movie_reviews: {len(mr)} criticas, {lab.count('pos')} positivas")
    print(f"gutenberg: {len(gutenberg_files())} libros")
    print(f"stopwords: {len(stopwords())} palabras")
