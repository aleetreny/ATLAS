"""Las ocho miniaturas de las tarjetas del modulo 7.

Cada una sale del JSON que su propio articulo publica, nunca de un calculo
nuevo, para que una tarjeta no pueda alejarse de la pagina que anuncia. Y cada
una lleva un `assert` sobre la propiedad que dibuja: adelgazar una figura para
que quepa en 400x300 cambia el dibujo, asi que si la miniatura ilustra algo
medido, hay que comprobar que lo sigue ilustrando despues de adelgazar.

Las reglas ya pagadas por las tandas anteriores y aplicadas aqui: una tarjeta
se mira un segundo a 460 px, asi que las marcas tienen que separarse; los
circulos se agrupan por color para escribir el `fill` una vez; las lineas van
en un solo `<path>`; las coordenadas se redondean a entero; y el fichero se
queda en la banda de 1 a 10 kB, porque la portada carga las cuarenta y tantas a
la vez.

Ejecutar desde cualquier sitio: `python src/utils/generate_web_thumbnails_7.py`.
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
    "confounding": "#202c47",
    "propensity": "#5c2938",
    "instruments": "#056605",
    "do-calculus": "#643c85",
    "conformal": "#945b43",
    "bayesian-nets": "#381a66",
    "attribution": "#473a04",
    "saliency": "#853c6f",
}
COSMOS = "#df2a5d"
AQUA = "#007faa"
SMILE = "#ff9900"


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


def polyline(points, colour, width=2, opacity=1.0, dash=None):
    d = " ".join(f"{'M' if i == 0 else 'L'}{round(x)},{round(y)}"
                 for i, (x, y) in enumerate(points))
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}"'
            f' opacity="{opacity}"{da}/>')


def dots(points, colour, r=3, opacity=0.8):
    """Agrupados por color: un `fill` para todos y un `use` por punto sale mas
    caro que escribir los circulos, pero un solo `<g>` con el fill heredado los
    deja sin atributo propio, que es de donde salia el fichero de 48 kB del
    modulo 4."""
    inner = "".join(f'<circle cx="{round(x)}" cy="{round(y)}" r="{r}"/>' for x, y in points)
    return f'  <g fill="{colour}" opacity="{opacity}">{inner}</g>'


def fit(vals, lo, hi, flip=False):
    a, b = min(vals), max(vals)
    if b - a < 1e-12:
        b = a + 1
    if flip:
        return lambda v: hi - (hi - lo) * (v - a) / (b - a)
    return lambda v: lo + (hi - lo) * (v - a) / (b - a)


# ------------------------------------------------------------- confounding
def thumb_confounding():
    """La inversion, que es la unica figura del articulo que se lee sin texto:
    una nube en tres colores, una recta que sube y tres que bajan."""
    d = load("confounding", "confounding.json")
    B = d["wine"]["best"]
    m = 16
    fx = fit(B["x"], m, W - m)
    fy = fit(B["y"], m + 8, H - m, flip=True)
    out = head(["la inversion de Simpson en el vino: la recta de todos sube y",
                "las tres de cada cultivar bajan"])
    colours = [ACCENT["confounding"], COSMOS, AQUA]
    for k in range(3):
        pts = [(fx(x), fy(y)) for x, y, g in zip(B["x"], B["y"], B["g"]) if g == k]
        out.append(dots(pts, colours[k], r=3, opacity=0.75))

    def line_for(xs, ys, colour, width):
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = sum((a - mx) ** 2 for a in xs)
        s = num / den
        lo, hi = min(xs), max(xs)
        return s, polyline([(fx(lo), fy(my + s * (lo - mx))),
                            (fx(hi), fy(my + s * (hi - mx)))], colour, width)

    s_all, path_all = line_for(B["x"], B["y"], INK, 3.2)
    out.append(path_all)
    withins = []
    for k in range(3):
        xs = [x for x, g in zip(B["x"], B["g"]) if g == k]
        ys = [y for y, g in zip(B["y"], B["g"]) if g == k]
        s, p = line_for(xs, ys, colours[k], 2.4)
        withins.append(s)
        out.append(p)
    # Lo que la miniatura afirma: una recta sube y las tres bajan. Si un dia
    # deja de ser verdad, el dibujo miente y el fichero no se escribe.
    assert all(s * s_all < 0 for s in withins), "la miniatura ya no ensena una inversion"
    write("confounding", out)


# -------------------------------------------------------------- propensity
def thumb_propensity():
    """Los dos histogramas de score, uno hacia arriba y otro hacia abajo: el
    solapamiento es la condicion de la que vive el articulo entero."""
    d = load("propensity", "propensity.json")
    C = d["cloud"]
    bins = 22
    top = [0] * bins
    bot = [0] * bins
    for e, x in zip(C["e_hat"], C["X"]):
        k = min(bins - 1, int(e * bins))
        (top if x == 1 else bot)[k] += 1
    m = 14
    mid = H / 2
    bw = (W - 2 * m) / bins
    hi = max(max(top), max(bot))
    out = head(["la distribucion del score en tratados y no tratados: se solapan",
                "en todo el rango, que es lo que el metodo necesita"])
    for k in range(bins):
        ht = (mid - m) * top[k] / hi
        hb = (mid - m) * bot[k] / hi
        x = m + k * bw
        out.append(f'  <rect x="{round(x)}" y="{round(mid - ht)}" width="{round(bw - 1)}" '
                   f'height="{round(ht)}" fill="{ACCENT["propensity"]}" opacity="0.9"/>')
        out.append(f'  <rect x="{round(x)}" y="{round(mid)}" width="{round(bw - 1)}" '
                   f'height="{round(hb)}" fill="{SMILE}" opacity="0.9"/>')
    out.append(f'  <path d="M{m},{round(mid)} L{W - m},{round(mid)}" stroke="{INK}" '
               f'stroke-width="1.2" fill="none"/>')
    both = sum(1 for a, b in zip(top, bot) if a > 0 and b > 0)
    assert both >= bins // 2, "los dos grupos ya no se solapan y la miniatura miente"
    write("propensity", out)


# ------------------------------------------------------------- instruments
def thumb_instruments():
    """El intervalo de efectos compatibles con los mismos datos, con la
    estimacion dentro: la figura que resume el articulo."""
    d = load("instruments", "instruments.json")
    Wd = d["worlds"]
    m = 30
    lo, hi = Wd["beta_lo"], Wd["beta_hi"]
    pad = (hi - lo) * 0.18
    fx = fit([lo - pad, hi + pad], m, W - m)
    y0, y1 = 110, 190
    out = head(["todos los efectos compatibles con la misma matriz de covarianzas,",
                "con lo que el estimador contesta marcado dentro"])
    out.append(f'  <rect x="{round(fx(lo))}" y="{y0}" width="{round(fx(hi) - fx(lo))}" '
               f'height="{y1 - y0}" fill="{ACCENT["instruments"]}" opacity="0.20"/>')
    for v in (lo, hi):
        out.append(f'  <path d="M{round(fx(v))},{y0} L{round(fx(v))},{y1}" '
                   f'stroke="{ACCENT["instruments"]}" stroke-width="3" fill="none"/>')
    out.append(f'  <path d="M{round(fx(Wd["iv_a"]))},{y0 - 34} '
               f'L{round(fx(Wd["iv_a"]))},{y1 + 34}" stroke="{COSMOS}" stroke-width="3" '
               f'stroke-dasharray="7 5" fill="none"/>')
    out.append(f'  <path d="M{m},{y1 + 46} L{W - m},{y1 + 46}" stroke="{INK}" '
               f'stroke-width="1.2" opacity="0.5" fill="none"/>')
    assert hi - lo > 1.0, "el intervalo se ha estrechado y la figura deja de decir nada"
    write("instruments", out)


# ------------------------------------------------------------- do-calculus
def thumb_docalculus():
    """Los dieciseis conjuntos de ajuste ordenados por error, con los dos que
    cumplen el criterio en el acento y los otros catorce apagados."""
    d = load("do-calculus", "docalculus.json")
    rows = sorted([r for r in d["exact"]["rows"] if r["error"] is not None],
                  key=lambda r: r["error"])
    m = 18
    bw = (W - 2 * m) / len(rows)
    top, bottom = 26, H - 26
    hi = max(r["error"] for r in rows)
    out = head(["los dieciseis conjuntos de ajuste posibles ordenados por error,",
                "y los dos que cumplen el criterio en cero"])
    for k, r in enumerate(rows):
        ht = (bottom - top) * (r["error"] / hi)
        out.append(f'  <rect x="{round(m + k * bw)}" y="{round(bottom - ht)}" '
                   f'width="{round(bw - 2)}" height="{round(max(ht, 2))}" '
                   f'fill="{ACCENT["do-calculus"] if r["backdoor"] else COSMOS}" '
                   f'opacity="{0.95 if r["backdoor"] else 0.55}"/>')
    out.append(f'  <path d="M{m},{bottom} L{W - m},{bottom}" stroke="{INK}" '
               f'stroke-width="1.2" fill="none"/>')
    valid = [r for r in rows if r["backdoor"]]
    assert valid and max(r["error"] for r in valid) < 1e-10, \
        "algun conjunto valido dejo de dar la respuesta exacta"
    write("do-calculus", out)


# ---------------------------------------------------------------- conformal
def thumb_conformal():
    """La Beta de la cobertura con la mitad que se queda por debajo de lo
    prometido sombreada: la figura que corrige la lectura ingenua."""
    d = load("conformal", "conformal.json")
    Hh = d["exact"]["hist"]
    alpha = d["meta"]["alpha"]
    xs, dens = Hh["x"], Hh["beta"]
    m = 18
    fx = fit(xs, m, W - m)
    fy = fit([0] + dens, m + 10, H - m, flip=True)
    out = head(["la cobertura que da cada calibracion, con la parte que se queda",
                "por debajo de lo prometido sombreada"])
    below = [(x, y) for x, y in zip(xs, dens) if x <= 1 - alpha]
    if below:
        pts = [(fx(below[0][0]), fy(0))] + [(fx(x), fy(y)) for x, y in below] \
            + [(fx(below[-1][0]), fy(0))]
        dd = " ".join(f"{'M' if i == 0 else 'L'}{round(x)},{round(y)}"
                      for i, (x, y) in enumerate(pts))
        out.append(f'  <path d="{dd}Z" fill="{COSMOS}" opacity="0.30"/>')
    out.append(polyline([(fx(x), fy(y)) for x, y in zip(xs, dens)],
                        ACCENT["conformal"], 3))
    out.append(f'  <path d="M{round(fx(1 - alpha))},{m} L{round(fx(1 - alpha))},{H - m}" '
               f'stroke="{INK}" stroke-width="2" stroke-dasharray="6 4" fill="none"/>')
    row = [r for r in d["exact"]["rows"] if r["n"] == Hh["n"]][0]
    assert row["below"] > 0.25, "la banda sombreada dejaria de significar nada"
    write("conformal", out)


# ----------------------------------------------------------- bayesian-nets
def thumb_bayesnets():
    """La banda predictiva exacta, con el hueco de los datos en medio: se ve
    que la incertidumbre se abre donde no hay puntos y no en los extremos."""
    d = load("bayesian-nets", "bayesnets.json")
    D = d["data"]
    E = d["exact"]
    m = 16
    fx = fit(D["xs"], m, W - m)
    lo = min(min(a - b for a, b in zip(E["pred_mean"], E["pred_sd"])), min(D["y"]))
    hi = max(max(a + b for a, b in zip(E["pred_mean"], E["pred_sd"])), max(D["y"]))
    fy = fit([lo, hi], m + 6, H - m, flip=True)
    up = [(fx(x), fy(mu + sd)) for x, mu, sd in zip(D["xs"], E["pred_mean"], E["pred_sd"])]
    dn = [(fx(x), fy(mu - sd)) for x, mu, sd in zip(D["xs"], E["pred_mean"], E["pred_sd"])]
    band = " ".join(f"{'M' if i == 0 else 'L'}{round(x)},{round(y)}"
                    for i, (x, y) in enumerate(up + dn[::-1]))
    out = head(["la banda de la posterior exacta: se abre en el hueco entre los",
                "datos y se cierra otra vez fuera de ellos"])
    out.append(f'  <path d="{band}Z" fill="{ACCENT["bayesian-nets"]}" opacity="0.22"/>')
    out.append(polyline([(fx(x), fy(v)) for x, v in zip(D["xs"], E["pred_mean"])],
                        ACCENT["bayesian-nets"], 2.6))
    out.append(dots([(fx(x), fy(y)) for x, y in zip(D["x"], D["y"])], INK, r=3.4, opacity=0.85))
    assert E["regions"]["in_gap"] > 2 * E["regions"]["on_data"], \
        "la banda ya no se abre en el hueco, que es lo unico que esta figura ensena"
    write("bayesian-nets", out)


# -------------------------------------------------------------- attribution
def thumb_attribution():
    """La cascada de una prediccion: diez barras que suman exactamente el hueco
    entre la media y esta botella."""
    d = load("attribution", "attribution.json")
    row = d["answer_key"]["rows"][0]
    base = d["answer_key"]["base"]
    order = sorted(range(len(row["phi"])), key=lambda j: -abs(row["phi"][j]))
    m = 20
    steps = []
    cur = base
    for j in order:
        steps.append((cur, cur + row["phi"][j], row["phi"][j]))
        cur += row["phi"][j]
    vals = [base, row["pred"]] + [s for a, b, _ in steps for s in (a, b)]
    fx = fit(vals, m, W - m)
    top, bh = 24, (H - 44) / len(steps)
    out = head(["la cascada de una prediccion: cada barra es el valor de Shapley",
                "exacto de una columna, y las diez suman el hueco entero"])
    for k, (a, b, v) in enumerate(steps):
        y = top + k * bh
        out.append(f'  <rect x="{round(fx(min(a, b)))}" y="{round(y)}" '
                   f'width="{round(max(abs(fx(b) - fx(a)), 2))}" height="{round(bh - 4)}" '
                   f'fill="{ACCENT["attribution"] if v >= 0 else COSMOS}" opacity="0.9"/>')
    for v, colour in ((base, INK), (row["pred"], COSMOS)):
        out.append(f'  <path d="M{round(fx(v))},{top - 8} L{round(fx(v))},{H - 14}" '
                   f'stroke="{colour}" stroke-width="2" stroke-dasharray="6 4" fill="none"/>')
    total = sum(row["phi"])
    assert abs(total - (row["pred"] - base)) < 1e-5, \
        "las barras dejaron de sumar la prediccion y la cascada no cerraria"
    write("attribution", out)


# ----------------------------------------------------------------- saliency
def thumb_saliency():
    """El marcador de los mapas contra la mascara verdadera, con los dos
    controles en otro color: la figura dice que hay controles."""
    d = load("saliency", "saliency.json")
    rows = sorted(d["scores"]["rows"], key=lambda r: -r["pointing"])
    m = 18
    bw = (W - 2 * m) / len(rows)
    top, bottom = 30, H - 30
    out = head(["cuanto acierta cada mapa el objeto de verdad, con los dos",
                "controles (la entrada sola y el ruido) en otro color"])
    controls = {"input", "random"}
    for k, r in enumerate(rows):
        ht = (bottom - top) * r["pointing"]
        out.append(f'  <rect x="{round(m + k * bw)}" y="{round(bottom - ht)}" '
                   f'width="{round(bw - 6)}" height="{round(max(ht, 2))}" '
                   f'fill="{COSMOS if r["key"] in controls else ACCENT["saliency"]}" '
                   f'opacity="{0.6 if r["key"] in controls else 0.95}"/>')
    out.append(f'  <path d="M{m},{bottom} L{W - m},{bottom}" stroke="{INK}" '
               f'stroke-width="1.2" fill="none"/>')
    best_control = max(r["pointing"] for r in rows if r["key"] in controls)
    best_method = max(r["pointing"] for r in rows if r["key"] not in controls)
    assert best_method > best_control, \
        "ningun metodo bate a sus controles y la figura contaria otra cosa"
    write("saliency", out)


def main():
    print("miniaturas del modulo 7:")
    for fn in (thumb_confounding, thumb_propensity, thumb_instruments, thumb_docalculus,
               thumb_conformal, thumb_bayesnets, thumb_attribution, thumb_saliency):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"  (falta {e.filename}, se salta)")


if __name__ == "__main__":
    main()
