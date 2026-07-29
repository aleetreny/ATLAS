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
import math
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
    accent = "#591c5c"
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
    accent = "#456b99"
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


# ---------------------------------------------------------------------------
def thumb_annealing_swarm():
    """El anillo de energias con las visitas de la cadena que enfria debajo."""
    accent = "#142a85"
    d = load("annealing-swarm", "anneal")
    e = d["boltzmann"]["energy"]
    path = d["reference"]["path"]
    n = len(e)
    counts = [0] * n
    for s in path:
        counts[s] += 1
    top = max(counts)
    pad, base = 26, 262
    sx = (W - 2 * pad) / (n - 1)
    hi = max(e)
    ey = lambda v: round(base - 132 * (v / hi) - 66)
    curve = " ".join(f"{round(pad + i * sx)},{ey(v)}" for i, v in enumerate(e))
    bars = "".join(
        f'<rect x="{round(pad + i * sx - sx * 0.42)}" '
        f'y="{round(base - 84 * (c / top))}" width="{max(2, round(sx * 0.84))}" '
        f'height="{round(84 * (c / top))}" />'
        for i, c in enumerate(counts) if c)
    # La miniatura existe para ensenar que la cadena se concentra donde la
    # energia es baja: si eso deja de ser verdad, la imagen deja de decirlo.
    low = sorted(range(n), key=lambda i: e[i])[: n // 4]
    share = sum(counts[i] for i in low) / sum(counts)
    assert share > 0.5, f"la cadena solo pasa el {share:.0%} en el cuarto mas bajo"
    body = (f'  <g fill="{accent}" fill-opacity="0.3">{bars}</g>\n'
            f'  <polyline points="{curve}" fill="none" stroke="{accent}" '
            f'stroke-width="3" />\n'
            f'  <line x1="{pad}" y1="{base}" x2="{W - pad}" y2="{base}" '
            f'stroke="{GREY}" stroke-width="1.5" />\n')
    write("annealing-swarm", body)


# ---------------------------------------------------------------------------
def thumb_bandits():
    """Las curvas de arrepentimiento de las cinco politicas."""
    accent = "#0e2d5c"
    d = load("bandits", "bandits")
    P = d["policies"]
    stride = P["greedy"]["stride"]
    T = d["meta"]["horizon"]
    top = max(P[k]["final"] for k in P)
    pad_l, pad_r, pad_t, pad_b = 34, 24, 30, 34
    px = lambda t: round(pad_l + (W - pad_l - pad_r) * t / T)
    py = lambda v: round(H - pad_b - (H - pad_t - pad_b) * v / (top * 1.05))
    curves = []
    for k, col in ((k, accent) for k in P):
        pts = P[k]["regret"]
        pts = " ".join(f"{px(i * stride)},{py(v)}" for i, v in enumerate(pts))
        curves.append((k, pts))
    worst = max(P, key=lambda k: P[k]["final"])
    bestk = min(P, key=lambda k: P[k]["final"])
    # La miniatura ensena que las politicas se separan: si dos curvas acabaran
    # juntas no diria nada.
    assert P[worst]["final"] / P[bestk]["final"] > 1.5, "las curvas no se separan"
    faint = "".join(f'<polyline points="{pts}" />' for k, pts in curves
                    if k not in (worst, bestk))
    body = (f'  <g fill="none" stroke="{accent}" stroke-width="2.5" '
            f'opacity="0.35">{faint}</g>\n'
            + "".join(f'  <polyline points="{pts}" fill="none" stroke="{accent}" '
                      f'stroke-width="4" />\n'
                      for k, pts in curves if k in (worst, bestk))
            + f'  <line x1="{pad_l}" y1="{H - pad_b}" x2="{W - pad_r}" '
              f'y2="{H - pad_b}" stroke="{GREY}" stroke-width="1.5" />\n')
    write("bandits", body)


# ---------------------------------------------------------------------------
def thumb_thompson_ucb():
    """Las cinco posteriores de Thompson tras mil tirones."""
    accent = "#994591"
    d = load("thompson-ucb", "ucb")
    snap = d["snapshots"]["thompson"][str(d["snapshots"]["marks"][-1])]
    alphas, betas = snap["alpha"], snap["beta"]

    def log_gamma(z):
        # Lanczos, suficiente para dibujar cinco densidades
        g = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
             -176.61502916214059, 12.507343278686905, -0.13857109526572012,
             9.9843695780195716e-06, 1.5056327351493116e-07]
        if z < 0.5:
            return math.log(math.pi / math.sin(math.pi * z)) - log_gamma(1 - z)
        x = z - 1
        a = 0.99999999999980993
        tt = x + 7.5
        for i, gi in enumerate(g):
            a += gi / (x + i + 1)
        return 0.5 * math.log(2 * math.pi) + (x + 0.5) * math.log(tt) - tt + math.log(a)

    def pdf(x, a, b):
        if x <= 0 or x >= 1:
            return 0.0
        lb = log_gamma(a) + log_gamma(b) - log_gamma(a + b)
        return math.exp((a - 1) * math.log(x) + (b - 1) * math.log(1 - x) - lb)

    pad, base = 28, 268
    n = 160
    curves, tops = [], []
    for a, b in zip(alphas, betas):
        ys = [pdf(i / n, a, b) for i in range(n + 1)]
        tops.append(max(ys))
        curves.append(ys)
    top = max(tops)
    px = lambda i: round(pad + (W - 2 * pad) * i / n)
    py = lambda v: round(base - (base - 34) * v / top)
    # La miniatura ensena que una posterior domina a las demas. Se afirma
    # contra la mediana y no contra la segunda, porque dos maquinas parecidas
    # tienen posteriores parecidas a proposito, y ademas se exige que la que
    # domina sea la de la mejor maquina y no otra.
    best = d["meta"]["arms"].index(max(d["meta"]["arms"]))
    assert tops.index(max(tops)) == best, "la posterior mas alta no es la de la mejor maquina"
    assert max(tops) / sorted(tops)[len(tops) // 2] > 1.4, "las posteriores salen iguales"
    body = ""
    for k, ys in enumerate(curves):
        pts = " ".join(f"{px(i)},{py(v)}" for i, v in enumerate(ys))
        lead = tops[k] == top
        body += (f'  <polyline points="{pts}" fill="none" stroke="{accent}" '
                 f'stroke-width="{4 if lead else 2}" opacity="{1 if lead else 0.45}" />\n')
    body += (f'  <line x1="{pad}" y1="{base}" x2="{W - pad}" y2="{base}" '
             f'stroke="{GREY}" stroke-width="1.5" />\n')
    write("thompson-ucb", body)


# ---------------------------------------------------------------------------
def thumb_q_learning():
    """El acantilado con los dos caminos aprendidos."""
    accent = "#99171d"
    d = load("q-learning", "qlearn")
    T = d["truth"]
    rows, cols = T["rows"], T["cols"]
    pad = 24
    cw = (W - 2 * pad) / cols
    ch = (H - 2 * pad - 40) / rows
    top = pad + 20
    cx = lambda rc: round(pad + rc[1] * cw + cw / 2)
    cy = lambda rc: round(top + rc[0] * ch + ch / 2)
    cells = "".join(
        f'<rect x="{round(pad + c * cw)}" y="{round(top + r * ch)}" '
        f'width="{round(cw) - 2}" height="{round(ch) - 2}" />'
        for r in range(rows) for c in range(cols))
    cliff = "".join(
        f'<rect x="{round(pad + c * cw)}" y="{round(top + r * ch)}" '
        f'width="{round(cw) - 2}" height="{round(ch) - 2}" />' for r, c in T["cliff"])
    q = d["learners"]["qlearning"]["path"]
    s = d["learners"]["sarsa"]["path"]
    # La miniatura existe para ensenar que los dos caminos son distintos.
    assert len(s) > len(q), "los dos caminos salieron iguales"
    line = lambda path: " ".join(f"{cx(p)},{cy(p)}" for p in path)
    body = (f'  <g fill="#ffffff" stroke="#d4dada" stroke-width="1.5">{cells}</g>\n'
            f'  <g fill="#df2a5d" fill-opacity="0.5">{cliff}</g>\n'
            f'  <polyline points="{line(s)}" fill="none" stroke="#003181" '
            f'stroke-width="5" stroke-linejoin="round" opacity="0.85" />\n'
            f'  <polyline points="{line(q)}" fill="none" stroke="{accent}" '
            f'stroke-width="5" stroke-linejoin="round" />\n')
    write("q-learning", body)


# ---------------------------------------------------------------------------
def thumb_dqn():
    """El acantilado sombreado con lo que la red se cree, y su camino."""
    accent = "#5c540e"
    d = load("dqn", "dqn")
    M = d["meta"]
    rows, cols = M["rows"], M["cols"]
    Q = d["weights"]["Q"]
    V = [max(row) for row in Q]
    lo, hi = min(V), max(V)
    pad = 24
    cw = (W - 2 * pad) / cols
    ch = (H - 2 * pad - 40) / rows
    top = pad + 20
    cx = lambda rc: round(pad + rc[1] * cw + cw / 2)
    cy = lambda rc: round(top + rc[0] * ch + ch / 2)

    def shade(v):
        f = (v - lo) / (hi - lo) if hi > lo else 0.5
        a = int(231 + (255 - 231) * f)
        b = int(223 + (255 - 223) * f)
        c = int(240 + (255 - 240) * f)
        return f"#{a:02x}{b:02x}{c:02x}"

    cells = "".join(
        f'<rect x="{round(pad + c * cw)}" y="{round(top + r_ * ch)}" '
        f'width="{round(cw) - 2}" height="{round(ch) - 2}" fill="{shade(V[r_ * cols + c])}" />'
        for r_ in range(rows) for c in range(cols))
    cliff = "".join(
        f'<rect x="{round(pad + c * cw)}" y="{round(top + r_ * ch)}" '
        f'width="{round(cw) - 2}" height="{round(ch) - 2}" />' for r_, c in M["cliff"])
    path = d["weights"]["path"]
    assert len(path) > 5, "el camino de la red no llega a ningun sitio"
    line = " ".join(f"{cx(p)},{cy(p)}" for p in path)
    body = (f'  <g stroke="#d4dada" stroke-width="1.5">{cells}</g>\n'
            f'  <g fill="#df2a5d" fill-opacity="0.45">{cliff}</g>\n'
            f'  <polyline points="{line}" fill="none" stroke="{accent}" '
            f'stroke-width="5" stroke-linejoin="round" />\n')
    write("dqn", body)


# ---------------------------------------------------------------------------
def thumb_policy_gradient():
    """Las dos curvas de error contra el gradiente exacto, con y sin linea base."""
    accent = "#647032"
    d = load("policy-gradient", "pg")
    G = d["gradient"]
    rows = {k: [r_ for r_ in G["rows"] if r_["kind"] == k] for k in ("plain", "baseline")}
    for a, b in zip(rows["plain"], rows["baseline"]):
        assert b["error"] < a["error"], f"la linea base no gana en {a['episodes']} episodios"
    eps = [r_["episodes"] for r_ in rows["plain"]]
    errs = [r_["error"] for r_ in rows["plain"]] + [r_["error"] for r_ in rows["baseline"]]
    pad, top, bot = 34, 40, 52
    lx = lambda e: round(pad + (math.log(e) / math.log(eps[-1])) * (W - 2 * pad))
    lo, hi = math.log(min(errs) * 0.8), math.log(max(errs) * 1.25)
    ly = lambda v: round(top + (hi - math.log(v)) / (hi - lo) * (H - top - bot))
    ref = ly(G["exact_norm"])
    body = (f'  <line x1="{pad}" y1="{ref}" x2="{W - pad}" y2="{ref}" stroke="{GREY}" '
            f'stroke-width="2" stroke-dasharray="7 6" />\n')
    for kind, colour, width in (("plain", GREY, 5), ("baseline", accent, 6)):
        pts = " ".join(f"{lx(r_['episodes'])},{ly(r_['error'])}" for r_ in rows[kind])
        body += (f'  <polyline points="{pts}" fill="none" stroke="{colour}" '
                 f'stroke-width="{width}" stroke-linejoin="round" />\n')
        dots = "".join(f'<circle cx="{lx(r_["episodes"])}" cy="{ly(r_["error"])}" r="7" />'
                       for r_ in rows[kind][::2])
        body += f'  <g fill="{colour}">{dots}</g>\n'
    return write("policy-gradient", body)


# ---------------------------------------------------------------------------
def thumb_mcts():
    """El tablero resuelto con las columnas que empatan y las que pierden, y el
    reparto de simulaciones de la busqueda encima."""
    accent = "#856314"
    d = load("mcts", "mcts")
    T = d["tree"]
    grid = T["grid"]
    rows, cols = len(grid), len(grid[0])
    exact = T["frames"][0]["exact"]
    frame = [f for f in T["frames"] if f["sims"] == max(g["sims"] for g in T["frames"])][0]
    total = sum(frame["counts"].values())
    assert total > 0, "la busqueda del fichero no visito nada"
    assert frame["pick"] in T["best"], "la miniatura ensena una busqueda que se equivoca"
    # casillas cuadradas: con celdas apaisadas el tablero no se lee como un
    # tablero, y una tarjeta se mira un segundo
    cell, top = 48, 12
    pad = round((W - cell * cols) / 2)
    cells, discs = [], []
    for r_ in range(rows):
        for c in range(cols):
            x, y = pad + c * cell, top + r_ * cell
            cells.append(f'<rect x="{x}" y="{y}" width="{cell - 4}" height="{cell - 4}" />')
            v = grid[r_][c]
            if v:
                discs.append((v, round(x + (cell - 4) / 2), round(y + (cell - 4) / 2)))
    base = H - 16
    tall = base - (top + rows * cell) - 26
    bars = []
    for c in range(cols):
        share = frame["counts"].get(str(c), 0) / total
        hgt = max(round(share * tall / max(frame["counts"].values()) * total), 2)
        good = exact[str(c)] == max(exact.values())
        bars.append(f'<rect x="{pad + c * cell + 6}" y="{base - hgt}" '
                    f'width="{cell - 16}" height="{hgt}" '
                    f'fill="{accent if good else "#c9ced0"}" />')
    body = (f'  <g fill="#e7eaec">{"".join(cells)}</g>\n'
            + "".join(f'  <circle cx="{x}" cy="{y}" r="15" '
                      f'fill="{accent if v == T["who"] else "#1c3f6e"}" />\n'
                      for v, x, y in discs)
            + f'  <g>{"".join(bars)}</g>\n')
    return write("mcts", body)


THUMBS = {
    "linear-programming": thumb_linear_programming,
    "genetic-algorithms": thumb_genetic_algorithms,
    "annealing-swarm": thumb_annealing_swarm,
    "bandits": thumb_bandits,
    "thompson-ucb": thumb_thompson_ucb,
    "q-learning": thumb_q_learning,
    "dqn": thumb_dqn,
    "policy-gradient": thumb_policy_gradient,
    "mcts": thumb_mcts,
}


def main():
    print("miniaturas del modulo 5:")
    for name, fn in THUMBS.items():
        fn()


if __name__ == "__main__":
    main()
