"""The four thumbnails for the new module 2.2 cards, drawn from the articles'
own measured data rather than from an illustrator's idea of them.

Each one is the picture the article's own widget draws, reduced to the size of a
card: the wine's map with PCA's rings on top of it, the Swiss roll with the
edges that ruin it, the digits under t-SNE, and the plane an autoencoder's two
numbers live in.

A card is about 460 px wide and the reader looks at it for a second, so every
picture here is thinned until the marks are separable at that size: a few
hundred dots at r = 4, not two thousand at r = 2.4. The thinning is also what
keeps these files in the 2-8 kB band the other thirty thumbnails live in, which
matters because the front page loads all of them at once.

Run from anywhere: `python src/utils/generate_web_thumbnails_2_2.py`.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "thumbnails"
W, H = 400, 300
BG = "#f1f3f3"


def fit(P, pad=26, w=W, h=H):
    """Square units, centred, filling the card."""
    P = np.asarray(P, dtype=float)
    lo, hi = P.min(0), P.max(0)
    c = (lo + hi) / 2
    span = (hi - lo).max() * 1.04
    unit = min((w - 2 * pad) / span, (h - 2 * pad) / span)
    return np.column_stack([
        w / 2 + (P[:, 0] - c[0]) * unit,
        h / 2 - (P[:, 1] - c[1]) * unit,
    ])


def head():
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
            f'  <rect width="{W}" height="{H}" fill="{BG}" />']


def dots(P, colours, r, opacity=None, per_line=3, order=None):
    """Circles grouped by fill, so the colour is written once per group.

    Grouping is also what fixes the draw order, and for the roll that matters:
    the groups are the turns of the spiral, so drawing them from the inside out
    is what makes the sheet look solid rather than transparent.
    """
    op = "" if opacity is None else f' opacity="{opacity}"'
    out = [f'  <g stroke="{BG}" stroke-width="0.6"{op}>']
    for col in (order if order is not None else sorted(set(colours))):
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


def segments(P, pairs, **attrs):
    """Many lines as one path, which costs a third of what <line> elements do."""
    d = "".join(f"M{P[i,0]:.0f} {P[i,1]:.0f}L{P[j,0]:.0f} {P[j,1]:.0f}" for i, j in pairs)
    bits = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return [f'  <path d="{d}" fill="none" {bits} />']


def write(name, lines):
    p = OUT / f"{name}.svg"
    p.write_text("\n".join(lines + ["</svg>", ""]), encoding="utf-8")
    print(f"  wrote {p.name} ({p.stat().st_size / 1024:.1f} kB)")


# ------------------------------------------------------------------- distances
# The picture is the article's opening claim: scaling that never sees a column
# lands on the same map PCA draws from all thirteen of them. So the rings have
# to be a real PCA of the matrix, not the same coordinates drawn twice.
d = json.loads((ROOT / "distances" / "data" / "distances.json").read_text("utf-8"))
X = np.array(d["wine"]["matrix"], dtype=float)
cls = d["wine"]["classes"]

D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(2)
B = -0.5 * (D2 - D2.mean(1, keepdims=True) - D2.mean(0, keepdims=True) + D2.mean())
w_, V = np.linalg.eigh(B)
o = np.argsort(-w_)
MDS = V[:, o[:2]] * np.sqrt(np.maximum(w_[o[:2]], 0))

Xc = X - X.mean(0)
_, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
PCA = Xc @ Vt[:2].T
for k in range(2):                     # an eigenvector's sign is arbitrary in both
    if np.dot(PCA[:, k], MDS[:, k]) < 0:
        PCA[:, k] *= -1
gap = float(np.abs(PCA - MDS).max())
assert gap < 1e-8, f"scaling and PCA disagree by {gap}, which the article says they cannot"

S = fit(MDS)
Sp = fit(PCA)                          # the same transform, since the inputs coincide
COL = ["#7e22ce", "#007faa", "#ff9900"]
lines = head()
lines.append("  <!-- dots: scaling, which only ever saw the distances -->")
lines += dots(S, [COL[c] for c in cls], 4.4)
lines.append("  <!-- rings: PCA on the thirteen columns, landing on top of them -->")
lines.append('  <g fill="none" stroke="#232f3e" stroke-width="1.3" opacity="0.8">')
row = []
for (x, y) in Sp[::3]:
    row.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="8" />')
    if len(row) == 3:
        lines.append("    " + "".join(row))
        row = []
if row:
    lines.append("    " + "".join(row))
lines.append("  </g>")
write("distances", lines)
print(f"  (PCA and classical scaling agree to {gap:.2e})")

# ------------------------------------------------------------------- manifolds
m = json.loads((ROOT / "manifolds" / "data" / "manifolds.json").read_text("utf-8"))
Pm = np.array(m["roll"]["points"], dtype=float)
CH = np.array(m["roll"]["chart"], dtype=float)
KEEP = 2                               # every second point: enough to read as a sheet
Pm, CH = Pm[::KEEP], CH[::KEEP]
# The article's widget looks at the roll from (0.62, 0.42), which works at 780 px
# with a reader who can study it. Shrunk to a card that view is a coloured blob,
# so the card looks almost straight down the height axis instead and gets the
# spiral itself. Same points, same graph, different camera.
a, b = 0.0, 1.30
proj = np.column_stack([
    Pm[:, 0] * np.cos(a) + Pm[:, 2] * np.sin(a),
    -(Pm[:, 1] * np.cos(b)) + (-Pm[:, 0] * np.sin(a) + Pm[:, 2] * np.cos(a)) * np.sin(b),
])
S = fit(proj, pad=18)
K = 5
Dm = np.sqrt(((Pm[:, None, :] - Pm[None, :, :]) ** 2).sum(2))
Gm = np.sqrt(((CH[:, None, :] - CH[None, :, :]) ** 2).sum(2))
idx = np.argsort(Dm, axis=1)[:, 1:K + 1]
edges = sorted({(min(i, int(j)), max(i, int(j))) for i in range(len(Pm)) for j in idx[i]})
shorts = [(i, j) for i, j in edges if Gm[i, j] > 2.0 * Dm[i, j]]
assert shorts, f"no short circuit at k = {K} on every {KEEP}th point, so the picture has no subject"


def turbo(t):
    """A short, honest stand-in for d3's turbo: the same ends, the same middle."""
    stops = [(0.19, 0.07, 0.23), (0.09, 0.60, 0.87), (0.55, 0.93, 0.35),
             (0.98, 0.72, 0.15), (0.48, 0.02, 0.01)]
    t = min(max(t, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    f = t - i
    c = [stops[i][k] * (1 - f) + stops[i + 1][k] * f for k in range(3)]
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c)


lo, hi = CH[:, 0].min(), CH[:, 0].max()
BANDS = 10                             # a ramp read in ten steps, so ten fill groups
band = np.clip(((CH[:, 0] - lo) / (hi - lo) * BANDS).astype(int), 0, BANDS - 1)
ramp = [turbo((k + 0.5) / BANDS) for k in range(BANDS)]
cols = [ramp[k] for k in band]

lines = head()
lines.append("  <!-- the neighbourhood graph, and the edges that jump between turns -->")
short_set = set(shorts)
lines += segments(S, [e for e in edges if e not in short_set],
                  stroke="#232f3e", stroke_width="0.45", opacity="0.28")
lines += segments(S, shorts, stroke="#9a3412", stroke_width="2.4")
lines.append("  <!-- the sheet, drawn from the innermost turn outwards -->")
lines += dots(S, cols, 3.4, order=ramp)
write("manifolds", lines)
print(f"  ({len(shorts)} short circuit edges among {len(edges)} at k = {K})")

# ------------------------------------------------------------------ neighbours
n = json.loads((ROOT / "neighbours" / "data" / "neighbours.json").read_text("utf-8"))
STEP = 10
Zt = np.array(n["big"]["tsne"], dtype=float)[::STEP]
lab = np.array(n["big"]["labels"])[::STEP]
S = fit(Zt, pad=16)
TEN = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
       "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
lines = head()
lines.append(f"  <!-- every {STEP}th of the 2,000 digits, placed by t-SNE, coloured by which digit -->")
lines += dots(S, [TEN[c] for c in lab], 4.4, opacity=0.92)
write("neighbours", lines)
print(f"  ({len(S)} digits shown)")

# ---------------------------------------------------------------- autoencoders
ae = json.loads((ROOT / "autoencoders" / "data" / "autoencoders.json").read_text("utf-8"))
Za = np.array(ae["net"]["latent"], dtype=float)
laba = np.array(ae["net"]["labels"])
STEP = max(1, len(Za) // 220)
Za, laba = Za[::STEP], laba[::STEP]
S = fit(Za, pad=18)
cx, cy = float(np.median(S[:, 0])), float(np.median(S[:, 1]))
lines = head()
lines.append("  <!-- the plane the encoder's two numbers live in, and one point in it -->")
lines += dots(S, [TEN[c] for c in laba], 4.4, opacity=0.9)
lines.append(f'  <g stroke="#232f3e" stroke-width="1" opacity="0.5">')
lines.append(f'    <line x1="{cx:.0f}" y1="6" x2="{cx:.0f}" y2="{H - 6}" />')
lines.append(f'    <line x1="6" y1="{cy:.0f}" x2="{W - 6}" y2="{cy:.0f}" />')
lines.append("  </g>")
lines.append(f'  <circle cx="{cx:.0f}" cy="{cy:.0f}" r="9" fill="none" '
             f'stroke="#be123c" stroke-width="3.4" />')
write("autoencoders", lines)
print(f"  ({len(S)} digits shown)")
