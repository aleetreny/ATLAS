"""Las ocho miniaturas de las tarjetas del modulo 8.

Mismas reglas que las tandas anteriores y por los mismos motivos ya pagados:
cada figura sale del JSON que su propio articulo publica (nunca de un calculo
nuevo, para que la tarjeta no pueda alejarse de la pagina que anuncia), cada
una lleva un `assert` sobre la propiedad que dibuja, los circulos se agrupan por
color, las lineas van en un solo `<path>`, las coordenadas se redondean a entero
y el fichero se queda por debajo de los 12 kB.

Y una regla propia de esta tanda: **ninguna de las ocho lleva una linea de
referencia que sea el minimo del conjunto**, porque en el modulo 3.3 eso dejo
una linea pegada al eje que se leia como una segunda regla. En una tarjeta va
una afirmacion y una anotacion.

Ejecutar desde cualquier sitio: `python src/utils/generate_web_thumbnails_8.py`.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "thumbnails"
W, H = 400, 300
BG = "#f1f3f3"
INK = "#232f3e"

ACCENT = {
    "hyperparameters": "#171799",
    "automl": "#48522d",
    "distillation": "#a30d5d",
    "lora": "#790da3",
    "crossval": "#a3483e",
    "metrics": "#5c3257",
    "ann-search": "#3e6d70",
    "vector-db": "#854960",
}
COSMOS = "#df2a5d"
AQUA = "#007faa"
SMILE = "#ff9900"
ANCHOR = "#003181"


def head(comment):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
            f'  <rect width="{W}" height="{H}" fill="{BG}" />'] \
        + [f'  <!-- {c} -->' for c in comment]


def write(name, lines):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.svg"
    path.write_text("\n".join(lines) + "\n</svg>\n", encoding="utf-8")
    kb = path.stat().st_size / 1024
    print(f"  {path.name}: {kb:.1f} kB")
    assert kb < 12, f"{name} pesa {kb:.1f} kB, y la portada carga las ochenta a la vez"


def load(slug, fname):
    return json.loads((ROOT / slug / "data" / fname).read_text("utf-8"))


def polyline(points, colour, width=2, opacity=1.0, dash=None):
    d = " ".join(f"{'M' if i == 0 else 'L'}{round(x)},{round(y)}"
                 for i, (x, y) in enumerate(points))
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}"'
            f' opacity="{opacity}"{da}/>')


def dots(points, colour, r=3, opacity=0.85):
    inner = "".join(f'<circle cx="{round(x)}" cy="{round(y)}" r="{r}"/>' for x, y in points)
    return f'  <g fill="{colour}" opacity="{opacity}">{inner}</g>'


def fit(vals, lo, hi, flip=False):
    a, b = min(vals), max(vals)
    if b - a < 1e-12:
        b = a + 1
    if flip:
        return lambda v: hi - (hi - lo) * (v - a) / (b - a)
    return lambda v: lo + (hi - lo) * (v - a) / (b - a)


def logfit(vals, lo, hi, flip=False):
    import math
    ls = [math.log(v) for v in vals]
    f = fit(ls, lo, hi, flip)
    return lambda v: f(math.log(v))


def mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


# ----------------------------------------------------------- hyperparameters
def thumb_hyperparameters():
    """La superficie entera, que es lo que el articulo tiene y nadie mas: una
    rejilla de celdas puntuadas con la meseta de arriba marcada."""
    d = load("hyperparameters", "hyperparameters.json")
    S = d["surface"]
    n = len(S["cv"])
    floor = d["floor"]["cv_sd"]
    top = max(max(row) for row in S["cv"])
    lo = min(min(row) for row in S["cv"])
    m = 14
    cw = (W - 2 * m) / n
    ch = (H - 2 * m) / n
    out = head(["las 961 celdas de la superficie, con la meseta que ningun",
                "buscador puede distinguir de la mejor marcada en naranja"])
    near = 0
    cells = []
    for i, row in enumerate(S["cv"]):
        for j, v in enumerate(row):
            t = (v - lo) / max(top - lo, 1e-9)
            colour = mix(BG, ACCENT["hyperparameters"], t ** 0.75)
            x = m + j * cw
            y = H - m - (i + 1) * ch
            cells.append(f'<rect x="{round(x)}" y="{round(y)}" width="{round(cw) + 1}"'
                         f' height="{round(ch) + 1}" fill="{colour}"/>')
            if v >= top - floor:
                near += 1
    out.append("  " + "".join(cells))
    marks = []
    for i, row in enumerate(S["cv"]):
        for j, v in enumerate(row):
            if v < top - floor:
                continue
            x = m + j * cw
            y = H - m - (i + 1) * ch
            marks.append(f'<rect x="{round(x)}" y="{round(y)}" width="{round(cw)}"'
                         f' height="{round(ch)}"/>')
    out.append(f'  <g fill="none" stroke="{SMILE}" stroke-width="1.2">'
               + "".join(marks) + "</g>")
    assert 0 < near < n * n * 0.6, \
        f"la meseta ocupa {near} de {n * n} celdas y la tarjeta deja de decir nada"
    write("hyperparameters", out)


# ------------------------------------------------------------------- automl
def thumb_automl():
    """Cuanto sube el sesgo del ganador segun se buscan mas tuberias, una curva
    por tabla: la afirmacion es que todas suben."""
    d = load("automl", "automl.json")
    tables = list(d["tables"])
    bs = [r["b"] for r in d["tables"][tables[0]]["curse"]]
    gaps = [r["gap"] for t in tables for r in d["tables"][t]["curse"]]
    m = 22
    fx = logfit(bs, m + 6, W - m)
    fy = fit(gaps + [0.0], m, H - m - 6, flip=True)
    out = head(["el sesgo del mejor de b tuberias, una curva por tabla:",
                "cuanto mas se busca, mas exagera la puntuacion que gana"])
    out.append(polyline([(m, fy(0)), (W - m, fy(0))], INK, 1.2, 0.5, dash="4 4"))
    colours = [ACCENT["automl"], ANCHOR, COSMOS, AQUA]
    rises = 0
    for k, t in enumerate(tables):
        rows = d["tables"][t]["curse"]
        pts = [(fx(r["b"]), fy(r["gap"])) for r in rows]
        out.append(polyline(pts, colours[k % len(colours)], 2.6))
        if rows[-1]["gap"] > rows[0]["gap"]:
            rises += 1
    assert rises >= len(tables) - 1, \
        f"solo {rises} de {len(tables)} curvas suben y la tarjeta afirma que suben"
    write("automl", out)


# ------------------------------------------------------------- distillation
def thumb_distillation():
    """La curva de transferencia con su control: lo aprendido contra la misma
    arquitectura sin entrenar contra los pixeles."""
    d = load("distillation", "distillation.json")
    R = d["transfer"]["forward"]["rows"]
    ns = [r["n"] for r in R]
    vals = [r[k] for r in R for k in ("transfer", "random", "pixels")]
    m = 20
    fx = logfit(ns, m + 6, W - m)
    fy = fit(vals, m + 6, H - m, flip=True)
    out = head(["la curva de transferencia y su control: la misma red sin",
                "entrenar ya llega bastante lejos, y los pixeles no"])
    for key, colour, width in (("pixels", INK, 2.2), ("random", AQUA, 2.4),
                               ("transfer", ACCENT["distillation"], 3.0)):
        out.append(polyline([(fx(r["n"]), fy(r[key])) for r in R], colour, width))
        out.append(dots([(fx(r["n"]), fy(r[key])) for r in R], colour, r=3))
    first = R[0]
    assert first["transfer"] > first["random"] > first["pixels"], \
        "el orden de los tres brazos al principio ha cambiado y la tarjeta miente"
    write("distillation", out)


# --------------------------------------------------------------------- lora
def thumb_lora():
    """El espectro del cambio contra el de una matriz al azar de la misma norma,
    que es la unica forma de que 'rango bajo' signifique algo."""
    d = load("lora", "lora.json")
    L = d["spectra"]["layers"][0]
    n = min(len(L["delta"]["energy"]), len(L["random"]["energy"]))
    m = 22
    fx = fit(list(range(1, n + 1)), m + 6, W - m)
    fy = fit([0.0, 1.0], m + 6, H - m, flip=True)
    out = head(["la energia del cambio que hace el ajuste fino, contra la de una",
                "matriz al azar de la misma norma: la de arriba es la del cambio"])
    out.append(polyline([(m, fy(0.9)), (W - m, fy(0.9))], SMILE, 1.4, 0.9, dash="5 3"))
    for key, colour, width in (("random", INK, 2.2), ("base", AQUA, 2.0),
                               ("delta", ACCENT["lora"], 3.0)):
        e = L[key]["energy"][:n]
        out.append(polyline([(fx(i + 1), fy(v)) for i, v in enumerate(e)], colour, width))
    assert L["delta"]["r90"] < L["random"]["r90"], \
        (f"el cambio necesita {L['delta']['r90']} direcciones y el azar "
         f"{L['random']['r90']}: la tarjeta afirma lo contrario")
    write("lora", out)


# ----------------------------------------------------------------- crossval
def thumb_crossval():
    """Los dos histogramas del experimento sin senal: elegir columnas mirando
    todo, contra elegirlas dentro del pliegue, contra la unica respuesta."""
    d = load("crossval", "crossval.json")
    N = d["noise"]
    lo, hi = 0.35, 0.8
    bins = 26
    m = 22

    def hist(arr):
        out = [0] * bins
        for v in arr:
            k = min(bins - 1, max(0, int((v - lo) / (hi - lo) * bins)))
            out[k] += 1
        return out

    a = hist(N["outside_all"])
    b = hist(N["inside_all"])
    top = max(max(a), max(b))
    fx = fit([lo, hi], m, W - m)
    fy = fit([0, top], m + 10, H - m, flip=True)
    bw = (W - 2 * m) / bins
    out = head(["cien filas, cinco mil columnas de ruido y etiquetas a cara o",
                "cruz: la izquierda es la respuesta y la derecha es una fuga"])
    for arr, colour in ((b, ACCENT["crossval"]), (a, COSMOS)):
        rects = []
        for k, c in enumerate(arr):
            if not c:
                continue
            x = fx(lo) + k * bw
            y = fy(c)
            rects.append(f'<rect x="{round(x)}" y="{round(y)}" width="{round(bw) + 1}"'
                         f' height="{round(H - m - y)}"/>')
        out.append(f'  <g fill="{colour}" opacity="0.75">' + "".join(rects) + "</g>")
    out.append(polyline([(fx(0.5), m + 6), (fx(0.5), H - m)], INK, 1.6, 0.9, dash="5 3"))
    assert N["outside"] > N["inside"] + 0.05, \
        "los dos brazos ya no se separan y la tarjeta no ensena una fuga"
    write("crossval", out)


# ------------------------------------------------------------------ metrics
def thumb_metrics():
    """Las 176.851 matrices de confusion, F1 contra el coeficiente de
    correlacion: una nube ancha es una nube de desacuerdos."""
    d = load("metrics", "metrics.json")
    C = d["enumerate"]["cloud"]
    m = 22
    fx = fit([0.0, 1.0], m + 6, W - m)
    fy = fit([-1.0, 1.0], m, H - m, flip=True)
    out = head(["cada matriz de confusion de cien casos, F1 en horizontal y el",
                "coeficiente de correlacion en vertical: la anchura es el desacuerdo"])
    out.append(polyline([(m, fy(0)), (W - m, fy(0))], INK, 1.2, 0.5))
    plain = [(fx(f1), fy(mcc)) for tp, fp, f1, mcc in C if tp > 0]
    zero = [(fx(f1), fy(mcc)) for tp, fp, f1, mcc in C if tp == 0]
    out.append(dots(plain, ACCENT["metrics"], r=1.7, opacity=0.4))
    out.append(dots(zero, COSMOS, r=2.2, opacity=0.9))
    lo = min(mcc for _, _, _, mcc in C)
    hi = max(mcc for _, _, _, mcc in C)
    assert lo < -0.2 and hi > 0.2, \
        "la nube ya no cruza el cero y la tarjeta deja de ensenar un desacuerdo"
    write("metrics", out)


# --------------------------------------------------------------- ann-search
def thumb_annsearch():
    """Recall contra distancias calculadas, con la busqueda exhaustiva como
    punto de la derecha: la forma entera del tema en una figura."""
    d = load("ann-search", "annsearch.json")
    I = d["indices"]
    costs = [r["cost"] for k in ("hnsw", "ivf", "pq") for r in I[k]] + [I["brute_cost"]]
    m = 24
    fx = logfit(costs, m + 6, W - m - 10)
    fy = fit([0.0, 1.0], m + 6, H - m, flip=True)
    out = head(["recall contra distancias calculadas por consulta, con la",
                "busqueda exhaustiva como el punto naranja de la derecha"])
    for key, colour in (("ivf", ANCHOR), ("pq", COSMOS), ("hnsw", ACCENT["ann-search"])):
        rows = sorted(I[key], key=lambda r: r["cost"])
        pts = [(fx(r["cost"]), fy(r["recall"])) for r in rows]
        out.append(polyline(pts, colour, 2.8))
        out.append(dots(pts, colour, r=3.4))
    out.append(f'  <circle cx="{round(fx(I["brute_cost"]))}" cy="{round(fy(1.0))}"'
               f' r="6" fill="{SMILE}"/>')
    best = max(I["hnsw"], key=lambda r: r["recall"])
    cheap = min((r for r in I["hnsw"] if r["recall"] >= 0.9),
                key=lambda r: r["cost"], default=None)
    assert best["recall"] > 0.9 and cheap and cheap["cost"] < I["brute_cost"], \
        "el grafo ya no llega al 90% mas barato que mirarlo todo"
    write("ann-search", out)


# ---------------------------------------------------------------- vector-db
def thumb_vectordb():
    """Lo que un filtro le hace a un indice: el recall se hunde segun el filtro
    aprieta, y la alternativa exacta cuesta lo que cuesta."""
    d = load("vector-db", "vectordb.json")
    rows = sorted(d["filter"]["rows"], key=lambda r: r["selectivity"])
    sels = [r["selectivity"] for r in rows]
    m = 24
    fx = logfit(sels, m + 6, W - m)
    fy = fit([0.0, 1.0], m + 6, H - m, flip=True)
    out = head(["el recall dentro del filtro segun el filtro aprieta: buscar y",
                "luego descartar se hunde, pedir de mas lo sostiene"])
    out.append(polyline([(m, fy(1.0)), (W - m, fy(1.0))], SMILE, 1.6, 0.9, dash="5 3"))
    for key, colour, width in (("post_recall", COSMOS, 3.0),
                               ("over_recall", ACCENT["vector-db"], 2.6)):
        pts = [(fx(r["selectivity"]), fy(r[key])) for r in rows]
        out.append(polyline(pts, colour, width))
        out.append(dots(pts, colour, r=3.4))
    assert rows[0]["post_recall"] < rows[-1]["post_recall"] - 0.05, \
        "el recall ya no se hunde al apretar el filtro y la tarjeta miente"
    write("vector-db", out)


def main():
    for fn in (thumb_hyperparameters, thumb_automl, thumb_distillation, thumb_lora,
               thumb_crossval, thumb_metrics, thumb_annsearch, thumb_vectordb):
        fn()


if __name__ == "__main__":
    main()
