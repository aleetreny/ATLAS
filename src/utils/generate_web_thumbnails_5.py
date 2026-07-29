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


THUMBS = {
    "linear-programming": thumb_linear_programming,
}


def main():
    print("miniaturas del modulo 5:")
    for name, fn in THUMBS.items():
        fn()


if __name__ == "__main__":
    main()
