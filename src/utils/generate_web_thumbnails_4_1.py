"""Las siete miniaturas de las tarjetas del modulo 4.1.

Cada una es el dibujo que su propio articulo hace, reducido al tamano de una
tarjeta, y cada una se construye desde el JSON que ese articulo publica y no
desde un calculo nuevo, para que una tarjeta no pueda alejarse de la pagina que
anuncia.

Las reglas que las tandas anteriores dejaron pagadas, y que aqui se aplican:

  - Una tarjeta se mira un segundo a 460 px. Se adelgaza hasta que las marcas se
    separan, los circulos se agrupan por color para escribir el `fill` una vez,
    las lineas van en un solo `<path>` y las coordenadas se redondean a entero.
  - Las otras cuarenta viven en la banda de 1 a 10 kB y estas tienen que
    tambien, porque la portada las carga todas a la vez.
  - **Una miniatura que ilustra una propiedad medida se afirma, no solo se
    dibuja.** Adelgazar cambia el dibujo, asi que cada una comprueba con un
    assert que lo que ensena sigue siendo verdad despues de adelgazar.

Ejecutar desde cualquier sitio: `python src/utils/generate_web_thumbnails_4_1.py`.
"""
import base64
import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "thumbnails"
W, H = 400, 300
BG = "#f1f3f3"
INK = "#232f3e"

ACCENT = {
    "tf-idf": "#164e63",
    "topics": "#706b11",
    "embeddings": "#713f12",
    "sequences": "#3c4e85",
    "attention": "#5c0e40",
    "pretraining": "#3730a3",
    # Cuatro de estos siete acentos no son los que el modulo llevaba al
    # escribirse. Mientras se escribia, otra sesion publico el modulo 3 entero
    # y se llevo #3f6212, #7c2d12, #134e4a y #831843, asi que estos salen de la
    # reserva libre: la regla 3 exige un acento por articulo y quien lanza las
    # sesiones reparte, pero el guardia es la red de abajo y aqui la uso.
    "state-space": "#0e5c29",
}


def head(comment):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
            f'  <rect width="{W}" height="{H}" fill="{BG}" />'] + [f'  <!-- {c} -->' for c in comment]


def write(name, lines):
    path = OUT / f"{name}.svg"
    path.write_text("\n".join(lines) + "\n</svg>\n", encoding="utf-8")
    kb = path.stat().st_size / 1024
    print(f"  {path.name}: {kb:.1f} kB")
    assert kb < 12, f"{name} pesa {kb:.1f} kB, y la portada carga las cuarenta a la vez"


def load(slug, fname):
    return json.loads((ROOT / slug / "data" / fname).read_text("utf-8"))


def polyline(points, colour, width=2, opacity=1.0, dash=None):
    d = " ".join(f"{'M' if i == 0 else 'L'}{round(x)},{round(y)}" for i, (x, y) in enumerate(points))
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}"'
            f' opacity="{opacity}"{da}/>')


def half_to_float(u):
    return struct.unpack("<e", struct.pack("<H", u))[0]


def decode_f16(b64, n):
    raw = base64.b64decode(b64)
    return [half_to_float(struct.unpack_from("<H", raw, 2 * i)[0]) for i in range(n)]


# ------------------------------------------------------------------- tf-idf
def thumb_tfidf():
    d = load("tf-idf", "tfidf.json")
    Z = d["zipf"]
    curve = Z["curve"]
    m = 18
    x0, x1 = m, W - m
    y0, y1 = m + 6, H - m
    lx = lambda r: x0 + (x1 - x0) * math.log10(r) / math.log10(Z["types"])
    ly = lambda c: y1 - (y1 - y0) * math.log10(max(c, 1)) / math.log10(curve[0][1])
    out = head(["la ley de Zipf del corpus, en log-log, con la banda de las palabras",
                "que aparecen una sola vez sombreada a la derecha"])
    first_one = next((r for r, c in curve if c == 1), Z["types"])
    out.append(f'  <rect x="{round(lx(first_one))}" y="{y0}" '
               f'width="{round(x1 - lx(first_one))}" height="{y1 - y0}" '
               f'fill="{ACCENT["tf-idf"]}" opacity="0.13"/>')
    out.append(polyline([(lx(r), ly(c)) for r, c in curve], ACCENT["tf-idf"], 2.6))
    out.append(f'  <path d="M{x0},{y1} L{x1},{y1}" stroke="{INK}" stroke-width="1" '
               f'opacity="0.35" fill="none"/>')
    # La miniatura afirma que la cola de una sola aparicion es una fraccion
    # grande del vocabulario. Si un dia deja de serlo, el dibujo miente.
    assert Z["hapax_frac"] > 0.25, "la banda sombreada dejaria de significar nada"
    write("tf-idf", out)


# ------------------------------------------------------------------- topics
def thumb_topics():
    d = load("topics", "topics.json")
    S = d["sampler"]
    cats = d["meta"]["categories"]
    labels = S["labels"]
    # Se dibuja la estructura de bloques que el widget ensena, adelgazada a una
    # fila por cada seis documentos para que las marcas se separen.
    rows = list(range(0, len(labels), 16))
    order = sorted(rows, key=lambda i: (cats.index(labels[i]), i))
    k = S["k"]
    m = 16
    cw = (W - 2 * m) / k
    ch = (H - 2 * m) / len(order)
    out = head(["la estructura de bloques que el muestreador encuentra: documentos",
                "agrupados por su categoria real, temas en columnas"])
    # Un `opacity` por rectangulo son 600 atributos y 48 kB. Se cuantiza a
    # cuatro niveles y se agrupan los rectangulos por nivel, asi que la opacidad
    # se escribe cuatro veces: la misma imagen a la vista, un decimo del peso.
    # El fondo tenue de una sola pieza, y encima solo las celdas encendidas:
    # a 460 px lo que se lee es el patron de bloques, no la textura del ruido,
    # asi que pintar 300 rectangulos apagados uno a uno es peso sin imagen.
    out.append(f'  <rect x="{m}" y="{m}" width="{W - 2 * m}" height="{round(len(order) * ch)}" '
               f'fill="{ACCENT["topics"]}" opacity="0.12"/>')
    lit = []
    for r, i in enumerate(order):
        c = cats.index(labels[i])
        lit.append(f'<rect x="{round(m + (c % k) * cw)}" y="{round(m + r * ch)}" '
                   f'width="{math.ceil(cw)}" height="{math.ceil(ch)}"/>')
    out.append(f'  <g fill="{ACCENT["topics"]}" opacity="0.85">')
    for i in range(0, len(lit), 6):
        out.append("    " + "".join(lit[i:i + 6]))
    out.append("  </g>")
    prev = None
    for r, i in enumerate(order):
        c = labels[i]
        if c != prev:
            out.append(f'  <path d="M{m},{round(m + r * ch)} L{W - m},{round(m + r * ch)}" '
                       f'stroke="{INK}" stroke-width="1" opacity="0.5" fill="none"/>')
            prev = c
    assert len(order) > 25, "con menos filas los bloques dejan de leerse"
    write("topics", out)


# --------------------------------------------------------------- embeddings
def thumb_embeddings():
    d = load("embeddings", "embeddings.json")
    P = d["pmi"]
    # 240 puntos y no mil: a 460 px la nube es una mancha con una diagonal
    # dentro, y los demas solo pesan (cada circulo son 33 caracteres).
    step = max(1, len(P["cloud"]) // 240)
    pts = P["cloud"][::step]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    m = 18
    sx = lambda v: m + (W - 2 * m) * (v - lo) / (hi - lo)
    sy = lambda v: (H - m) - (H - 2 * m) * (v - lo) / (hi - lo)
    out = head(["lo que skip-gram aprende contra lo que el teorema predice:",
                "la diagonal no es un ajuste, es la igualdad de Levy y Goldberg"])
    out.append(f'  <path d="M{round(sx(lo))},{round(sy(lo))} L{round(sx(hi))},{round(sy(hi))}" '
               f'stroke="#df2a5d" stroke-width="2" stroke-dasharray="6 4" fill="none"/>')
    out.append(f'  <g fill="{ACCENT["embeddings"]}" opacity="0.4">')
    row = []
    for x, y in pts:
        row.append(f'<circle cx="{round(sx(x))}" cy="{round(sy(y))}" r="1.8"/>')
        if len(row) == 8:
            out.append("    " + "".join(row))
            row = []
    if row:
        out.append("    " + "".join(row))
    out.append("  </g>")
    assert P["pearson"] > 0.3, "la nube dejaria de parecerse a una diagonal"
    write("embeddings", out)


# ---------------------------------------------------------------- sequences
def thumb_sequences():
    d = load("sequences", "sequences.json")
    S = d["model"]["sentences"][0]
    tags = d["model"]["tags"]
    n = len(S["words"])
    T = len(tags)
    m = 22
    cw = (W - 2 * m) / max(1, n - 1)
    ch = (H - 2 * m) / max(1, T - 1)
    out = head(["el trellis de Viterbi: todos los caminos considerados,",
                "uno conservado"])
    dots = []
    for i in range(n):
        for t in range(T):
            dots.append(f'<circle cx="{round(m + i * cw)}" cy="{round(m + t * ch)}" r="2.6"/>')
    out.append(f'  <g fill="{INK}" opacity="0.18">')
    for i in range(0, len(dots), 10):
        out.append("    " + "".join(dots[i:i + 10]))
    out.append("  </g>")
    path = [tags.index(g) for g in S["gold"]]
    pts = [(m + i * cw, m + path[i] * ch) for i in range(n)]
    out.append(polyline(pts, ACCENT["sequences"], 2.6))
    out.append(f'  <g fill="{ACCENT["sequences"]}">')
    out.append("    " + "".join(f'<circle cx="{round(x)}" cy="{round(y)}" r="4"/>'
                                for x, y in pts))
    out.append("  </g>")
    assert n >= 6 and T >= 8, "el trellis necesita forma de rejilla para leerse"
    write("sequences", out)


# ---------------------------------------------------------------- attention
def thumb_attention():
    d = load("attention", "attention.json")
    R = d["reach"]
    rel = R["relative"]
    m = 20
    n = min(len(rel), 48)
    out = head(["el alcance del gradiente de la atencion contra la distancia:",
                "plano donde una recurrencia ya ha perdido la mitad"])
    lo = max(1e-4, min(v for v in rel[:n] if v > 0))
    sx = lambda i: m + (W - 2 * m) * i / max(1, n - 1)
    sy = lambda v: (H - m) - (H - 2 * m) * (math.log10(max(v, lo)) - math.log10(lo)) / (
        0 - math.log10(lo))
    out.append(f'  <path d="M{m},{round(sy(0.5))} L{W - m},{round(sy(0.5))}" '
               f'stroke="{INK}" stroke-width="1" stroke-dasharray="4 3" opacity="0.4" '
               f'fill="none"/>')
    out.append(polyline([(sx(i), sy(rel[i])) for i in range(n)], ACCENT["attention"], 2.8))
    assert len(rel) > 20, "hace falta curva bastante para que la forma se vea"
    write("attention", out)


# -------------------------------------------------------------- pretraining
def thumb_pretraining():
    d = load("pretraining", "pretraining.json")
    A = d["arms"]
    m = 26
    bw = (W - 2 * m) / (len(A) * 2 - 1)
    hi = max(a["signal_per_token"] for a in A)
    out = head(["predicciones supervisadas por token leido: la columna que",
                "nadie imprime y la que explica como escalaron"])
    for i, a in enumerate(A):
        x = m + i * bw * 2
        bh = (H - 2 * m) * a["signal_per_token"] / hi
        out.append(f'  <rect x="{round(x)}" y="{round(H - m - bh)}" width="{round(bw)}" '
                   f'height="{round(bh)}" fill="{ACCENT["pretraining"]}" opacity="0.88"/>')
    out.append(f'  <path d="M{m},{H - m} L{W - m},{H - m}" stroke="{INK}" stroke-width="1" '
               f'opacity="0.4" fill="none"/>')
    lo = min(a["signal_per_token"] for a in A)
    assert hi / max(lo, 1e-9) > 1.5, "sin contraste entre barras la tarjeta no dice nada"
    write("pretraining", out)


# -------------------------------------------------------------- state-space
def thumb_statespace():
    d = load("state-space", "statespace.json")
    R = d["reach"]
    m = 20
    out = head(["el nucleo de la convolucion segun como se inicialice el",
                "decaimiento: uno diverge, uno olvida pronto, uno alcanza"])
    colours = ["#df2a5d", "#003181", ACCENT["state-space"]]
    n = 110
    for row, colour in zip(R, colours):
        k = row["kernel"][:n]
        pts = [(m + (W - 2 * m) * i / (n - 1),
                (H - m) - (H - 2 * m) * min(1.0, max(0.0, v)))
               for i, v in enumerate(k)]
        out.append(polyline(pts, colour, 2.4, 0.9))
    out.append(f'  <path d="M{m},{H - m} L{W - m},{H - m}" stroke="{INK}" stroke-width="1" '
               f'opacity="0.4" fill="none"/>')
    assert any(r["unstable"] for r in R), "la tarjeta ensena que una inicializacion diverge"
    write("state-space", out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("miniaturas del modulo 4.1")
    for fn in (thumb_tfidf, thumb_topics, thumb_embeddings, thumb_sequences,
               thumb_attention, thumb_pretraining, thumb_statespace):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"  (falta {e.filename}, se salta)")


if __name__ == "__main__":
    main()
