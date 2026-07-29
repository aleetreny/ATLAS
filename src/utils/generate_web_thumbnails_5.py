"""Las miniaturas de las tarjetas del modulo 5 (agentes y estrategia).

Cada una sale del JSON que su propio articulo publica, no de un calculo nuevo,
para que una tarjeta no pueda alejarse de la pagina que anuncia. Y cada una se
afirma con un `assert` antes de escribirse: adelgazar un dibujo para que se lea
a 460 px puede cambiar lo que ensena, y eso ya paso una vez en el modulo 2.2.

Reglas heredadas de las tandas anteriores:
  - una tarjeta se mira un segundo a 460 px, asi que menos marcas y mas grandes
  - los circulos se agrupan por color para escribir el `fill` una vez, las
    lineas van en un solo `<path>` y las coordenadas se redondean a entero
  - las demas viven entre 1 y 10 kB y estas tambien, porque la portada las
    carga todas a la vez

Ejecutar desde cualquier sitio: `python src/utils/generate_web_thumbnails_5.py`.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "thumbnails"
OUT.mkdir(parents=True, exist_ok=True)
PAPER = "#f1f3f3"
GREY = "#8e9aa6"
W, H = 400, 300


def load(article, name):
    return json.loads((ROOT / article / "data" / f"{name}.json").read_text())


def write(name, body, cap_kb=10):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">\n'
           f'  <rect width="{W}" height="{H}" fill="{PAPER}" />\n{body}</svg>\n')
    f = OUT / f"{name}.svg"
    f.write_text(svg)
    kb = f.stat().st_size / 1024
    assert kb < cap_kb, f"{name}.svg pesa {kb:.1f} kB"
    print(f"  {name}.svg  {kb:.1f} kB")


# ---------------------------------------------------------------------------
def thumb_linear_programming():
    """El politopo, sus esquinas, y la esquina que gana."""
    accent = "#674599"
    d = load("linear-programming", "lp")
    verts = d["polytope"]["vertices"]
    best = d["polytope"]["best_index"]
    xs = [v["x"] for v in verts]
    ys = [v["y"] for v in verts]
    pad, top = 46, 40
    sx = (W - 2 * pad) / (max(xs) - min(xs))
    sy = (H - 2 * top) / (max(ys) - min(ys))
    px = lambda v: round(pad + (v - min(xs)) * sx)
    py = lambda v: round(H - top - (v - min(ys)) * sy)
    ring = " ".join(f"{px(v['x'])},{py(v['y'])}" for v in verts)
    # La recta de igual beneficio por la esquina ganadora. La longitud se fija
    # en pixeles y no en unidades del problema: los dos ejes se escalan distinto
    # para llenar la tarjeta, asi que una longitud en datos sale de la imagen.
    bx, by = verts[best]["x"], verts[best]["y"]
    c1, c2 = d["polytope"]["profit"]
    ux, uy = c2 * sy, c1 * sx          # perpendicular al objetivo, en pantalla
    n = (ux ** 2 + uy ** 2) ** 0.5
    L = 105
    line = (f'{round(px(bx) - ux / n * L)},{round(py(by) - uy / n * L)} '
            f'{round(px(bx) + ux / n * L)},{round(py(by) + uy / n * L)}')
    dots = "".join(f'<circle cx="{px(v["x"])}" cy="{py(v["y"])}" r="7" />'
                   for i, v in enumerate(verts) if i != best)
    assert len(verts) >= 5, "el politopo perdio esquinas al dibujarlo"
    body = (f'  <polygon points="{ring}" fill="{accent}" fill-opacity="0.16" '
            f'stroke="{accent}" stroke-width="2.5" />\n'
            f'  <polyline points="{line}" fill="none" stroke="{GREY}" '
            f'stroke-width="2.5" stroke-dasharray="9 6" />\n'
            f'  <g fill="{PAPER}" stroke="{accent}" stroke-width="2.5">{dots}</g>\n'
            f'  <circle cx="{px(bx)}" cy="{py(by)}" r="11" fill="{accent}" />\n')
    write("linear-programming", body)


# ---------------------------------------------------------------------------
def thumb_genetic_algorithms():
    """La poblacion final de la tirada de referencia, como bits."""
    accent = "#995e45"
    d = load("genetic-algorithms", "ga")
    rows = d["reference"]["final_population"][:10]
    bits = d["meta"]["bits"]
    block = d["meta"]["block"]
    pad_x, pad_y = 30, 34
    cw = (W - 2 * pad_x) / bits
    rh = (H - 2 * pad_y) / len(rows)
    # Las celdas encendidas van en UN path: cuatrocientos <rect> con su propio
    # atributo pesan 21 kB, que es cuarenta veces lo que pesa una miniatura del
    # sitio. El fondo apagado es una banda por fila, no una celda por bit.
    on = []
    n_on = 0
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            if not v:
                continue
            n_on += 1
            x = round(pad_x + j * cw)
            y = round(pad_y + i * rh)
            on.append(f"M{x},{y}h{round(cw)}v{round(rh) - 2}h-{round(cw)}z")
    share = n_on / (len(rows) * bits)
    # Si la miniatura sale casi toda de un color no ensena nada, y esta imagen
    # existe para ensenar que la poblacion ha convergido pero no del todo.
    assert 0.15 < share < 0.85, f"la poblacion sale {share:.2f} de unos"
    bands = "".join(
        f'<rect x="{pad_x}" y="{round(pad_y + i * rh)}" width="{W - 2 * pad_x}" '
        f'height="{round(rh) - 2}" />' for i in range(len(rows)))
    lines = "".join(
        f'<line x1="{round(pad_x + k * block * cw)}" y1="{pad_y - 8}" '
        f'x2="{round(pad_x + k * block * cw)}" y2="{H - pad_y + 6}" />'
        for k in range(bits // block + 1))
    body = (f'  <g fill="#e2e6e7">{bands}</g>\n'
            f'  <path d="{"".join(on)}" fill="{accent}" />\n'
            f'  <g stroke="{GREY}" stroke-width="1.2" opacity="0.7">{lines}</g>\n')
    write("genetic-algorithms", body)


THUMBS = {
    "linear-programming": thumb_linear_programming,
    "genetic-algorithms": thumb_genetic_algorithms,
}


def main():
    print("miniaturas del modulo 5:")
    for name, fn in THUMBS.items():
        fn()


if __name__ == "__main__":
    main()
