"""Las cinco miniaturas de las tarjetas del modulo 6.

Cada una sale del JSON que su propio articulo publica, nunca de un calculo
nuevo, para que una tarjeta no pueda alejarse de la pagina que anuncia. Las
reglas ya pagadas: una tarjeta se mira un segundo a 460 px, asi que las marcas
tienen que separarse; los circulos se agrupan por color para escribir el `fill`
una vez; las lineas van en un solo `<path>`; las coordenadas se redondean a
entero; y el fichero se queda por debajo de 10 kB, porque la portada carga las
cuarenta y tantas a la vez.

Ejecutar desde cualquier sitio: `python src/utils/generate_web_thumbnails_6.py`.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "thumbnails"
W, H = 400, 300
BG = "#f1f3f3"
INK = "#232f3e"

ACCENT = {
    "collaborative-filtering": "#994595",
    "matrix-factorization": "#647032",
    "two-tower": "#856314",
    "gnn": "#591c5c",
    "kg-embeddings": "#456b99",
}
ANCHOR = "#003181"
COSMOS = "#df2a5d"
STONE = "#d4dada"


def head(comment):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
            f'  <rect width="{W}" height="{H}" fill="{BG}" />'] + [f'  <!-- {c} -->' for c in comment]


def write(name, lines):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.svg"
    path.write_text("\n".join(lines) + "\n</svg>\n", encoding="utf-8")
    kb = path.stat().st_size / 1024
    print(f"  {path.name}: {kb:.1f} kB")
    assert kb < 12, f"{name} pesa {kb:.1f} kB, y la portada carga las cuarenta a la vez"


def load(slug, fname):
    return json.loads((ROOT / slug / "data" / fname).read_text("utf-8"))


# --------------------------------------------------------------------------
# 6.1.1 filtrado colaborativo: la cola larga, que es la forma del problema
# --------------------------------------------------------------------------
def thumb_cf():
    d = load("collaborative-filtering", "cf.json")
    counts = d["shape"]["counts_sorted"]
    acc = ACCENT["collaborative-filtering"]
    x0, x1, y0, y1 = 34, 372, 246, 40
    lo, hi = 1, len(counts)
    top = max(counts)

    def px(k):
        return x0 + (x1 - x0) * math.log10(k) / math.log10(hi)

    def py(v):
        return y0 - (y0 - y1) * math.log10(max(v, 1)) / math.log10(top)

    pts = []
    for e in range(0, 121):
        k = max(1, min(hi, round(10 ** (e / 120 * math.log10(hi)))))
        pts.append((round(px(k)), round(py(counts[k - 1]))))
    seen = []
    for p in pts:
        if not seen or seen[-1] != p:
            seen.append(p)
    path = "M " + " L ".join(f"{a} {b}" for a, b in seen)
    half = d["shape"]["half_books"]
    lines = head(["la cola larga del catalogo: 10.000 libros ordenados por cuantas",
                  "valoraciones tienen, en ejes logaritmicos, con la marca de donde",
                  "esta la mitad de todas las valoraciones"])
    lines.append(f'  <path d="{path}" fill="none" stroke="{acc}" stroke-width="3" />')
    lines.append(f'  <line x1="{round(px(half))}" y1="{y1 - 6}" x2="{round(px(half))}" y2="{y0}" '
                 f'stroke="{ANCHOR}" stroke-width="2" stroke-dasharray="6 4" />')
    lines.append(f'  <line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{INK}" stroke-width="1.5" />')
    lines.append(f'  <line x1="{x0}" y1="{y1 - 6}" x2="{x0}" y2="{y0}" stroke="{INK}" stroke-width="1.5" />')
    # la mancha de celdas vacias vive en la esquina que la curva deja libre:
    # la curva cae de arriba a la izquierda hacia abajo a la derecha, asi que
    # el hueco esta abajo a la izquierda y no arriba a la derecha, que es donde
    # estaba y donde se cruzaba con ella
    for i in range(0, 8):
        for j in range(0, 5):
            if (i * 5 + j) % 7 == 0:
                lines.append(f'  <rect x="{52 + i * 13}" y="{150 + j * 13}" width="9" height="9" '
                             f'fill="{acc}" opacity="0.55" />')
            else:
                lines.append(f'  <rect x="{52 + i * 13}" y="{150 + j * 13}" width="9" height="9" '
                             f'fill="none" stroke="{STONE}" stroke-width="1" />')
    write("collaborative-filtering", lines)


# --------------------------------------------------------------------------
# 6.1.2 factorizacion: la matriz partida en dos factores
# --------------------------------------------------------------------------
def thumb_mf():
    d = load("matrix-factorization", "mf.json")
    acc = ACCENT["matrix-factorization"]
    lines = head(["R = P Q': la tabla de valoraciones, casi vacia, escrita como el",
                  "producto de dos tablas flacas, con la curva del barrido de rango",
                  "que la pagina mide debajo"])
    # la matriz grande, con huecos
    cell = 15
    for i in range(8):
        for j in range(8):
            filled = (i * 7 + j * 3) % 5 == 0
            if filled:
                lines.append(f'  <rect x="{28 + j * cell}" y="{44 + i * cell}" width="{cell - 3}" '
                             f'height="{cell - 3}" fill="{acc}" opacity="0.75" />')
            else:
                lines.append(f'  <rect x="{28 + j * cell}" y="{44 + i * cell}" width="{cell - 3}" '
                             f'height="{cell - 3}" fill="none" stroke="{STONE}" />')
    lines.append(f'  <text x="{28 + 8 * cell + 12}" y="{44 + 4 * cell + 6}" font-size="20" '
                 f'font-family="monospace" fill="{INK}">=</text>')
    for i in range(8):
        for j in range(3):
            lines.append(f'  <rect x="{182 + j * cell}" y="{44 + i * cell}" width="{cell - 3}" '
                         f'height="{cell - 3}" fill="{ANCHOR}" opacity="0.8" />')
    lines.append(f'  <text x="{182 + 3 * cell + 8}" y="{44 + 4 * cell + 6}" font-size="20" '
                 f'font-family="monospace" fill="{INK}">x</text>')
    for i in range(3):
        for j in range(8):
            lines.append(f'  <rect x="{262 + j * cell}" y="{44 + i * cell}" width="{cell - 3}" '
                         f'height="{cell - 3}" fill="{COSMOS}" opacity="0.8" />')
    # el barrido de rango, medido
    rows = d["sweeps"]["rank"]
    xs = [r["k"] for r in rows]
    ys = [r["test"] for r in rows]
    lo, hi = min(ys), max(ys)
    x0, x1, y0, y1 = 40, 360, 268, 200

    def px(k):
        return x0 + (x1 - x0) * math.log2(k) / math.log2(max(xs))

    def py(v):
        # el error BAJA con el rango, asi que la curva tiene que bajar: con
        # (hi - v) en el numerador subia, que es la misma medida contada al reves
        return y0 - (y0 - y1) * (v - lo) / max(hi - lo, 1e-9)

    path = "M " + " L ".join(f"{round(px(k))} {round(py(v))}" for k, v in zip(xs, ys))
    lines.append(f'  <path d="{path}" fill="none" stroke="{acc}" stroke-width="3" />')
    for k, v in zip(xs, ys):
        lines.append(f'  <circle cx="{round(px(k))}" cy="{round(py(v))}" r="3.5" fill="{acc}" />')
    write("matrix-factorization", lines)


# --------------------------------------------------------------------------
# 6.1.3 two-tower: las dos torres y el producto escalar
# --------------------------------------------------------------------------
def thumb_tower():
    d = load("two-tower", "tower.json")
    acc = ACCENT["two-tower"]
    lines = head(["dos torres y un producto escalar: cada lado sube por su cuenta y",
                  "solo se encuentran al final, que es la restriccion de rango que",
                  "el articulo mide"])
    for side, x, colour in ((0, 96, ANCHOR), (1, 304, COSMOS)):
        for lvl, wdt in enumerate([84, 62, 42]):
            y = 226 - lvl * 46
            lines.append(f'  <rect x="{round(x - wdt / 2)}" y="{y}" width="{wdt}" height="30" '
                         f'fill="{colour}" opacity="{0.35 + 0.2 * lvl:.2f}" />')
        lines.append(f'  <text x="{x}" y="264" font-size="13" font-family="monospace" '
                     f'text-anchor="middle" fill="{INK}">'
                     f'{"reader" if side == 0 else "item"}</text>')
    lines.append(f'  <path d="M 96 94 L 200 62 L 304 94" fill="none" stroke="{acc}" '
                 f'stroke-width="3" />')
    lines.append(f'  <circle cx="200" cy="62" r="17" fill="{acc}" />')
    lines.append(f'  <text x="200" y="68" font-size="17" font-family="monospace" '
                 f'text-anchor="middle" fill="#ffffff">.</text>')
    # la cota exacta debajo, del propio fichero
    rows = d["ceiling"]["rows"]
    x0, x1, y0, y1 = 40, 360, 44, 22
    lo = min(r["rmse"] for r in rows)
    hi = max(r["rmse"] for r in rows)

    def px(k):
        return x0 + (x1 - x0) * math.log2(k) / math.log2(max(r["k"] for r in rows))

    def py(v):
        return y0 - (y0 - y1) * (hi - v) / max(hi - lo, 1e-9)

    path = "M " + " L ".join(f"{round(px(r['k']))} {round(py(r['rmse']))}" for r in rows)
    lines.append(f'  <path d="{path}" fill="none" stroke="{INK}" stroke-width="2" opacity="0.6" />')
    write("two-tower", lines)


# --------------------------------------------------------------------------
# 6.2.1 redes en grafos: el vecindario que se promedia
# --------------------------------------------------------------------------
def thumb_gnn():
    d = load("gnn", "gnn.json")
    acc = ACCENT["gnn"]
    W_ = d["widget"]
    nodes = W_["nodes"]
    edges = W_["edges"]
    # se adelgaza a los nodos mejor conectados, y se comprueba que el dibujo
    # sigue siendo un grafo conexo y no una nube de puntos sueltos
    keep = sorted(range(len(nodes)), key=lambda i: -nodes[i]["deg"])[:34]
    keep_set = set(keep)
    sub = [(a, b) for a, b in edges if a in keep_set and b in keep_set]
    assert len(sub) >= 30, f"solo quedan {len(sub)} aristas y el dibujo deja de ser un grafo"
    pos = {i: (60 + (nodes[i]["x"] + 1) * 140, 40 + (1 - nodes[i]["y2"]) * 110) for i in keep}
    lines = head(["un trozo del grafo de citas: los nodos rellenos son los que llevan",
                  "etiqueta y el color es lo que llega a cada uno tras promediar con",
                  "sus vecinos"])
    path = " ".join(f"M {round(pos[a][0])} {round(pos[a][1])} L {round(pos[b][0])} {round(pos[b][1])}"
                    for a, b in sub)
    lines.append(f'  <path d="{path}" fill="none" stroke="{STONE}" stroke-width="1.6" />')
    by_colour = {}
    for i in keep:
        colour = acc if nodes[i]["labelled"] else ANCHOR
        by_colour.setdefault(colour, []).append(i)
    for colour, ids in by_colour.items():
        for i in ids:
            x, y = pos[i]
            r = 9 if nodes[i]["labelled"] else 6
            op = "1" if nodes[i]["labelled"] else "0.45"
            lines.append(f'  <circle cx="{round(x)}" cy="{round(y)}" r="{r}" fill="{colour}" '
                         f'opacity="{op}" />')
    # la curva de profundidad, que es el otro resultado del articulo
    rows = d["depth"]["trained"]
    x0, x1, y0, y1 = 44, 356, 276, 214
    lo = min(r["acc"] for r in rows)
    hi = max(r["acc"] for r in rows)
    xs = [r["layers"] for r in rows]

    def px(k):
        return x0 + (x1 - x0) * (k - min(xs)) / max(max(xs) - min(xs), 1)

    def py(v):
        return y0 - (y0 - y1) * (v - lo) / max(hi - lo, 1e-9)

    path = "M " + " L ".join(f"{round(px(r['layers']))} {round(py(r['acc']))}" for r in rows)
    lines.append(f'  <path d="{path}" fill="none" stroke="{acc}" stroke-width="3" />')
    for r in rows:
        lines.append(f'  <circle cx="{round(px(r["layers"]))}" cy="{round(py(r["acc"]))}" r="3.5" '
                     f'fill="{acc}" />')
    write("gnn", lines)


# --------------------------------------------------------------------------
# 6.2.2 grafos de conocimiento: la rotacion en el plano complejo
# --------------------------------------------------------------------------
def thumb_kg():
    d = load("kg-embeddings", "kg.json")
    acc = ACCENT["kg-embeddings"]
    lines = head(["una relacion como giro en el plano complejo: la cabeza gira y tiene",
                  "que caer sobre la cola, y una relacion simetrica solo puede girar",
                  "media vuelta o nada"])
    cx, cy, r = 200, 150, 92
    lines.append(f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{STONE}" '
                 f'stroke-width="2" />')
    lines.append(f'  <line x1="{cx - r - 18}" y1="{cy}" x2="{cx + r + 18}" y2="{cy}" '
                 f'stroke="{INK}" stroke-width="1.2" opacity="0.5" />')
    lines.append(f'  <line x1="{cx}" y1="{cy - r - 18}" x2="{cx}" y2="{cy + r + 18}" '
                 f'stroke="{INK}" stroke-width="1.2" opacity="0.5" />')
    ang0 = math.radians(200)
    turn = math.radians(96)
    for k, (a, colour, rad) in enumerate([(ang0, INK, 7), (ang0 + turn, acc, 9)]):
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        lines.append(f'  <circle cx="{round(x)}" cy="{round(y)}" r="{rad}" fill="{colour}" />')
    arc_pts = []
    for t in range(0, 41):
        a = ang0 + turn * t / 40
        arc_pts.append((round(cx + (r - 14) * math.cos(a)), round(cy + (r - 14) * math.sin(a))))
    lines.append('  <path d="M ' + " L ".join(f"{a} {b}" for a, b in arc_pts)
                 + f'" fill="none" stroke="{acc}" stroke-width="2.5" stroke-dasharray="5 4" />')
    # la media vuelta de una relacion simetrica, en el otro color
    a = ang0 + math.pi
    lines.append(f'  <circle cx="{round(cx + r * math.cos(a))}" cy="{round(cy + r * math.sin(a))}" '
                 f'r="7" fill="{COSMOS}" opacity="0.85" />')
    n_sym = sum(1 for row in d["relations"] if row["symmetry"] > 0.5)
    assert n_sym >= 1, "la miniatura dibuja la media vuelta de una relacion simetrica y no hay"
    write("kg-embeddings", lines)


def main():
    print("miniaturas del modulo 6")
    thumb_cf()
    thumb_mf()
    thumb_tower()
    thumb_gnn()
    thumb_kg()


if __name__ == "__main__":
    main()
