"""The five thumbnails for the module 3.2 cards, drawn from the articles' own
measured data rather than from an illustrator's idea of them.

Each one is the picture its article's own widget draws, reduced to the size of a
card: the plane a variational encoder maps digits into, the codebook a discrete
model replaces points with, the landscape an energy model walks on, the grid a
flow warps, and the bar chart of what a picture costs in bits.

A card is about 460 px wide and the reader looks at it for a second, so every
picture here is thinned until the marks are separable at that size, which is
also what keeps these files in the 2-8 kB band the other thirty-two thumbnails
live in.

Run from anywhere: `python src/utils/generate_web_thumbnails_3_2.py`.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "assets" / "thumbnails"
W, H = 400, 300
BG = "#f1f3f3"
INK = "#232f3e"
DIGIT10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def head():
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
            f'  <rect width="{W}" height="{H}" fill="{BG}" />']


def fit(P, pad=24, w=W, h=H):
    """Square units, centred, filling the card. Square because every picture
    here is a plane and an ellipse drawn from a circle is a lie."""
    P = np.asarray(P, dtype=float)
    lo, hi = P.min(0), P.max(0)
    c = (lo + hi) / 2
    span = (hi - lo).max() * 1.06
    unit = min((w - 2 * pad) / span, (h - 2 * pad) / span)
    return np.column_stack([w / 2 + (P[:, 0] - c[0]) * unit,
                            h / 2 - (P[:, 1] - c[1]) * unit]), unit, c


def dots(P, colours, r, opacity=None, per_line=3):
    op = "" if opacity is None else f' opacity="{opacity}"'
    out = [f'  <g stroke="{BG}" stroke-width="0.6"{op}>']
    for col in sorted(set(colours)):
        out.append(f'    <g fill="{col}">')
        row = []
        for (x, y), c in zip(P, colours):
            if c != col:
                continue
            row.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r}" />')
            if len(row) == per_line:
                out.append("      " + "".join(row))
                row = []
        if row:
            out.append("      " + "".join(row))
        out.append("    </g>")
    out.append("  </g>")
    return out


def write(name, lines):
    p = OUT / f"{name}.svg"
    p.write_text("\n".join(lines + ["</svg>", ""]), encoding="utf-8")
    print(f"  wrote {p.name} ({p.stat().st_size / 1024:.1f} kB)")


def load(folder, name):
    p = ROOT / folder / "data" / name
    if not p.is_file():
        print(f"  skipping {folder}: {p.name} is not there yet")
        return None
    return json.loads(p.read_text("utf-8"))


# ------------------------------------------------------------------------- vae
# The picture is the article's own claim: the codes sit inside the gaussian the
# model was trained towards, which is the thing an autoencoder's codes do not do.
d = load("vae", "vae.json")
if d:
    Z = np.array(d["net"]["mu"], dtype=float)
    lab = d["net"]["labels"]
    keep = np.arange(0, len(Z), 9)
    P, unit, c = fit(Z[keep], pad=30)
    lines = head()
    lines.append("  <!-- the codes the variational encoder produces, one in four -->")
    lines += dots(P, [DIGIT10[lab[i]] for i in keep], 3.8, opacity=0.85, per_line=6)
    lines.append("  <!-- and the standard gaussian it was trained towards -->")
    for s in (1, 2):
        lines.append(f'  <circle cx="{W / 2 - c[0] * unit:.1f}" cy="{H / 2 + c[1] * unit:.1f}" '
                     f'r="{s * unit:.1f}" fill="none" stroke="{INK}" stroke-width="1.6" '
                     f'stroke-dasharray="6 5" opacity="0.85" />')
    write("vae", lines)

# ---------------------------------------------------------------------- vq-vae
# Points coloured by which entry replaced them, and the entries themselves: the
# whole article in one picture, because a codebook IS the partition it induces.
d = load("vq-vae", "vqvae.json")
if d:
    S = d["toy"]["showcase"]
    pts = np.array(S["points"], dtype=float)
    assign = S["assign"][str(S["k"])] if str(S["k"]) in S["assign"] else \
        S["assign"][sorted(S["assign"], key=int)[-1]]
    book = np.array(next(r for r in d["toy"]["rows"] if r["k"] == S["k"])["codebook"],
                    dtype=float)
    both = np.vstack([pts, book])
    Pall, unit, c = fit(both, pad=26)
    P, B = Pall[:len(pts)], Pall[len(pts):]
    warm = ["#9c3d14", "#c96a2a", "#7f1d1d", "#b45309", "#a16207",
            "#7c2d12", "#8a4b12", "#6d2f1f", "#a8410f", "#5f2412", "#92400e"]
    lines = head()
    lines.append("  <!-- points, coloured by the entry that replaced them -->")
    lines += dots(P[::3], [warm[a % len(warm)] for a in assign[::3]], 3.6, opacity=0.82, per_line=6)
    lines.append("  <!-- and the entries -->")
    lines.append(f'  <g fill="none" stroke="{INK}" stroke-width="1.8">')
    row = []
    for (x, y) in B:
        row.append(f'<path d="M{x - 4:.0f} {y:.0f}h8M{x:.0f} {y - 4:.0f}v8" />')
        if len(row) == 4:
            lines.append("    " + "".join(row))
            row = []
    if row:
        lines.append("    " + "".join(row))
    lines.append("  </g>")
    write("vq-vae", lines)

# ------------------------------------------------------------------------- ebm
# The data the energy is fitted to, with walkers on it: the article is about the
# chain, so the chain is what the card shows.
d = load("ebm", "ebm.json")
if d:
    sample = np.array(d["truth"]["sample"], dtype=float)
    ends = np.array(d["nonoise"]["endpoints"], dtype=float)
    both = np.vstack([sample, ends])
    Pall, unit, c = fit(both, pad=24)
    P, E = Pall[:len(sample)], Pall[len(sample):]
    lines = head()
    lines.append("  <!-- the density the energy is fitted to -->")
    lines += dots(P[::5], ["#8e9aa6"] * len(P[::5]), 3.0, opacity=0.85, per_line=6)
    lines.append("  <!-- where the walkers end up when the noise is taken away -->")
    lines += dots(E[::14], ["#5f2412"] * len(E[::14]), 2.8, opacity=0.95, per_line=6)
    write("ebm", lines)

# ----------------------------------------------------------------------- flows
# The warped grid, computed here from the flow's own exported weights, because
# the article's claim is about what that grid can and cannot do.
d = load("flows", "flows.json")
if d:
    import base64

    N = d["net"]
    raw = np.frombuffer(base64.b64decode(N["weights_b64"]), dtype=np.float16,
                        count=N["count"]).astype(np.float64)
    spec = N["layers"]
    parity = N["parity"]
    s_max = N["s_max"]

    def dense(sp, X):
        nin, nout = sp["shape"]
        Wm = raw[sp["w_offset"]:sp["w_offset"] + sp["w_size"]].reshape(nout, nin).T
        return X @ Wm + raw[sp["b_offset"]:sp["b_offset"] + sp["b_size"]]

    def inverse(Z):
        Y = np.array(Z, dtype=float)
        for i in range(len(parity) - 1, -1, -1):
            a = 0 if parity[i] == 0 else 1
            b = 1 - a
            h = np.tanh(dense(spec[3 * i], Y[:, a:a + 1]))
            h = np.tanh(dense(spec[3 * i + 1], h))
            o = dense(spec[3 * i + 2], h)
            s = np.tanh(o[:, :1]) * s_max
            X = np.empty_like(Y)
            X[:, a] = Y[:, a]
            X[:, b] = (Y[:, b:b + 1] - o[:, 1:2]).ravel() * np.exp(-s).ravel()
            Y = X
        return Y

    LINES, STEPS, SPAN = 11, 26, 2.5
    curves = []
    for i in range(LINES):
        u = -SPAN + 2 * SPAN * i / (LINES - 1)
        v = np.linspace(-SPAN, SPAN, STEPS)
        curves.append(inverse(np.c_[np.full(STEPS, u), v]))
        curves.append(inverse(np.c_[v, np.full(STEPS, u)]))
    allpts = np.vstack(curves)
    keep = np.abs(allpts).max(1) < 6
    Pall, unit, c = fit(allpts[keep], pad=18)
    # map every curve through the same transform the fit chose
    lo, hi = allpts[keep].min(0), allpts[keep].max(0)
    cc = (lo + hi) / 2
    span = (hi - lo).max() * 1.06
    unit = min((W - 36) / span, (H - 36) / span)

    def to_px(Q):
        return np.column_stack([W / 2 + (Q[:, 0] - cc[0]) * unit,
                                H / 2 - (Q[:, 1] - cc[1]) * unit])

    lines = head()
    lines.append("  <!-- a square grid in the gaussian, pushed through the flow -->")
    lines.append(f'  <g fill="none" stroke="#8a4b12" stroke-width="1.1" opacity="0.75">')
    for cv in curves:
        Q = to_px(cv)
        inside = (Q[:, 0] > -60) & (Q[:, 0] < W + 60) & (Q[:, 1] > -60) & (Q[:, 1] < H + 60)
        if inside.sum() < 2:
            continue
        pth = "M" + "L".join(f"{x:.0f} {y:.0f}" for x, y in Q[inside])
        lines.append(f'    <path d="{pth}" />')
    lines.append("  </g>")
    write("flows", lines)

# ------------------------------------------------------------------------ glow
# The article is a comparison of code lengths, so the card is the comparison.
d = load("glow", "glow.json")
if d:
    C = d["compress"]
    bars = [("raw file", C["raw"]), ("greys", C["histogram"]),
            ("per pixel", C["per_pixel"]), ("gzip", C["gzip"]),
            ("PNG", C["png"]), ("this flow", C["flow"])]
    left, right, top = 96, W - 20, 26
    bh = (H - top - 22) / len(bars)
    scale = (right - left) / max(v for _, v in bars)
    lines = head()
    lines.append("  <!-- bits per pixel, on the same held out images -->")
    for i, (name, v) in enumerate(bars):
        y = top + i * bh
        col = "#6d2f1f" if name == "this flow" else "#8e9aa6"
        end = left + v * scale
        lines.append(f'  <rect x="{left}" y="{y + 3:.0f}" width="{max(v * scale, 1):.0f}" '
                     f'height="{bh - 9:.0f}" fill="{col}" />')
        lines.append(f'  <text x="{left - 8}" y="{y + bh / 2 + 3:.0f}" text-anchor="end" '
                     f'font-family="monospace" font-size="13" fill="{INK}">{name}</text>')
        # the longest bar reaches the right margin, so its number goes inside it:
        # outside, a four character label at 12px monospace runs off the card
        inside = end + 6 + 30 > W
        anchor = ' text-anchor="end"' if inside else ''
        lines.append(f'  <text x="{(end - 6) if inside else (end + 6):.0f}" '
                     f'y="{y + bh / 2 + 3:.0f}"{anchor} '
                     f'font-family="monospace" font-size="12" '
                     f'fill="{"#ffffff" if inside else INK}" '
                     f'opacity="0.8">{v:.2f}</text>')
    write("glow", lines)

print("done")
